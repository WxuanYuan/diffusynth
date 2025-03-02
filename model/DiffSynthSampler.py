import numpy as np
import torch
from tqdm import tqdm


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # Convert numpy array to tensor on the same device as timesteps, then index and convert to float
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    # Add extra dimensions until the tensor has the same number of dimensions as broadcast_shape
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    # Expand tensor to match the broadcast shape
    return res.expand(broadcast_shape)


class DiffSynthSampler:
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, device=None, mute=False,
                 height=128, max_batchsize=16, max_width=256, channels=4, train_width=64, noise_strategy="repeat"):
        """
        Initialize the sampler with diffusion parameters and noise configuration.

        :param timesteps: Number of timesteps for diffusion.
        :param beta_start: Starting value for beta schedule.
        :param beta_end: Ending value for beta schedule.
        :param device: Torch device to use ('cuda' or 'cpu'). If None, auto-select.
        :param mute: If True, disable tqdm progress bar.
        :param height: The height of the generated images.
        :param max_batchsize: Maximum batch size.
        :param max_width: Maximum width of the generated images.
        :param channels: Number of channels in the images.
        :param train_width: The width used during training.
        :param noise_strategy: Strategy for generating noise ('repeat' or non-repeat).
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.height = height
        self.train_width = train_width
        self.max_batchsize = max_batchsize
        self.max_width = max_width
        self.channels = channels
        self.num_timesteps = timesteps
        self.timestep_map = list(range(self.num_timesteps))
        self.betas = np.array(np.linspace(beta_start, beta_end, self.num_timesteps), dtype=np.float64)
        self.respaced = False
        self.define_beta_schedule()
        self.CFG = 1.0
        self.mute = mute
        self.noise_strategy = noise_strategy

    def get_deterministic_noise_tensor_non_repeat(self, batchsize, width, reference_noise=None):
        """
        Generate a deterministic noise tensor without repeating segments.

        :param batchsize: Number of samples in the batch.
        :param width: The required width of the noise tensor.
        :param reference_noise: Optional precomputed noise to use.
        :return: A noise tensor cropped to [batchsize, channels, height, width] and None (placeholder).
        """

        if reference_noise is None:
            large_noise_tensor = torch.randn((self.max_batchsize, self.channels, self.height, self.max_width), device=self.device)
        else:
            assert reference_noise.shape == (batchsize, self.channels, self.height, self.max_width), "reference_noise shape mismatch"
            large_noise_tensor = reference_noise
        return large_noise_tensor[:batchsize, :, :, :width], None

    def get_deterministic_noise_tensor(self, batchsize, width, reference_noise=None):
        """
        Generate a deterministic noise tensor based on the selected noise strategy.

        :param batchsize: Number of samples.
        :param width: Desired noise width.
        :param reference_noise: Optional reference noise.
        :return: A noise tensor and any concatenation points (if applicable).
        """

        if self.noise_strategy == "repeat":
            noise, concat_points = self.get_deterministic_noise_tensor_repeat(batchsize, width, reference_noise=reference_noise)
            return noise, concat_points
        else:
            noise, concat_points = self.get_deterministic_noise_tensor_non_repeat(batchsize, width, reference_noise=reference_noise)
            return noise, concat_points


    def get_deterministic_noise_tensor_repeat(self, batchsize, width, reference_noise=None):
        """
        Generate a deterministic noise tensor by repeating parts of the training noise.

        This function uses a "repeat" strategy to create a noise tensor with the same length as the training data.
        Depending on the required width, it may insert repeated segments from the first part of the noise tensor.

        :param batchsize: Number of samples.
        :param width: The desired width of the noise tensor.
        :param reference_noise: Optional reference noise of shape [batchsize, channels, height, train_width].
        :return: A noise tensor and a list of indices marking the concatenation points.
        """
        # Generate noise with the same length as the training width
        if reference_noise is None:
            train_noise_tensor = torch.randn((self.max_batchsize, self.channels, self.height, self.train_width), device=self.device)
        else:
            assert reference_noise.shape == (batchsize, self.channels, self.height, self.train_width), "reference_noise shape mismatch"
            train_noise_tensor = reference_noise

        release_width = int(self.train_width * 1.0 / 4)
        first_part_width = self.train_width - release_width

        first_part = train_noise_tensor[:batchsize, :, :, :first_part_width]
        release_part = train_noise_tensor[:batchsize, :, :, -release_width:]

        # If the required width is less than or equal to the original training width,
        # remove the middle part of the first_part.
        if width <= self.train_width:
            _first_part_head_width = int((width - release_width) / 2)
            _first_part_tail_width = width - release_width - _first_part_head_width
            all_parts = [first_part[:, :, :, :_first_part_head_width], first_part[:, :, :, -_first_part_tail_width:], release_part]

            # Concatenate the parts along the width dimension (4th dimension)
            noise_tensor = torch.cat(all_parts, dim=3)

            # Record the positions where parts were concatenated
            concat_points = [0]
            for part in all_parts[:-1]:
                next_point = concat_points[-1] + part.size(3)
                concat_points.append(next_point)

            return noise_tensor, concat_points

        # If the required width is greater than the original training width,
        # repeatedly insert middle parts of first_part.
        else:
            repeats = (width - release_width) // first_part_width
            extra = (width - release_width) % first_part_width

            _repeat_first_part_head_width = int(first_part_width / 2)
            _repeat_first_part_tail_width = first_part_width - _repeat_first_part_head_width

            repeated_first_head_parts = [first_part[:, :, :, :_repeat_first_part_head_width] for _ in range(repeats)]
            repeated_first_tail_parts = [first_part[:, :, :, -_repeat_first_part_tail_width:] for _ in range(repeats)]

            # Calculate the start index to extract a middle part from first_part
            _middle_part_start_index = (first_part_width - extra) // 2
            middle_part = first_part[:, :, :, _middle_part_start_index: _middle_part_start_index + extra]

            all_parts = repeated_first_head_parts + [middle_part] + repeated_first_tail_parts + [release_part]

            # Concatenate all parts along the width dimension
            noise_tensor = torch.cat(all_parts, dim=3)

            # Record concatenation points for each part
            concat_points = [0]
            for part in all_parts[:-1]:
                next_point = concat_points[-1] + part.size(3)
                concat_points.append(next_point)

            return noise_tensor, concat_points

    def define_beta_schedule(self):
        """
        Define the beta schedule and precompute all necessary diffusion process constants.
        """

        assert self.respaced == False, "This schedule has already been respaced!"
        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def activate_classifier_free_guidance(self, CFG, unconditional_condition):
        """
        Activate classifier-free guidance (CFG) by setting the guidance scale and storing the unconditional condition.

        :param CFG: The classifier-free guidance scale.
        :param unconditional_condition: The condition used for unconditional guidance.
        """
        assert (
                   not unconditional_condition is None) or CFG == 1.0, "For CFG != 1.0, unconditional_condition must be available"
        self.CFG = CFG
        self.unconditional_condition = unconditional_condition

    def respace(self, use_timesteps=None):
        """
        Respace the beta schedule to use only a subset of timesteps.

        :param use_timesteps: A list of timesteps to keep.
        """
        if not use_timesteps is None:
            last_alpha_cumprod = 1.0
            new_betas = []
            self.timestep_map = []
            for i, _alpha_cumprod in enumerate(self.alphas_cumprod):
                if i in use_timesteps:
                    new_betas.append(1 - _alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = _alpha_cumprod
                    self.timestep_map.append(i)
            self.num_timesteps = len(use_timesteps)
            self.betas = np.array(new_betas)
            self.define_beta_schedule()
            self.respaced = True

    def generate_linear_noise(self, shape, variance=1.0, first_endpoint=None, second_endpoint=None):
        """
        Generate a noise tensor with linear interpolation between endpoints.

        :param shape: The desired shape of the noise tensor.
        :param variance: The target variance of the noise.
        :param first_endpoint: Optional starting endpoint tensor.
        :param second_endpoint: Optional ending endpoint tensor.
        :return: A noise tensor with linear interpolation.
        """
        assert shape[1] == self.channels, "shape[1] != self.channels"
        assert shape[2] == self.height, "shape[2] != self.height"
        noise = torch.empty(*shape, device=self.device)

        # Case 3: Both endpoints are provided, perform linear interpolation between them.
        if first_endpoint is not None and second_endpoint is not None:
            for i in range(shape[0]):
                alpha = i / (shape[0] - 1)  # interpolation coefficient
                noise[i] = alpha * second_endpoint + (1 - alpha) * first_endpoint
            return noise  # Return the interpolated result without further adjustment.
        else:
            # When only the first endpoint is provided.
            if first_endpoint is not None:
                noise[0] = first_endpoint
                if shape[0] > 1:
                    noise[1], _ = self.get_deterministic_noise_tensor(1, shape[3])[0]
            else:
                noise[0], _ = self.get_deterministic_noise_tensor(1, shape[3])[0]
                if shape[0] > 1:
                    noise[1], _ = self.get_deterministic_noise_tensor(1, shape[3])[0]

            # Generate subsequent noise points using a linear recurrence.
            for i in range(2, shape[0]):
                noise[i] = 2 * noise[i - 1] - noise[i - 2]

        # Adjust noise variance to match the target variance.
        current_var = noise.var()
        stddev_ratio = torch.sqrt(variance / current_var)
        noise = noise * stddev_ratio

        # If the first endpoint is provided, shift the noise so that the first element matches.
        if first_endpoint is not None:
            shift = first_endpoint - noise[0]
            noise += shift

        return noise

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        assert x_start.shape[1] == self.channels, "shape[1] != self.channels"
        assert x_start.shape[2] == self.height, "shape[2] != self.height"

        if noise is None:
            # noise = torch.randn_like(x_start)
            noise, _ = self.get_deterministic_noise_tensor(x_start.shape[0], x_start.shape[3])

        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    @torch.no_grad()
    def ddim_sample(self, model, x, t, condition=None, ddim_eta=0.0):
        """
        Perform a single DDIM sampling step.

        :param model: The neural network model used for predicting noise.
        :param x: The current image tensor.
        :param t: The current timestep tensor.
        :param condition: Optional condition for guided sampling.
        :param ddim_eta: The eta parameter controlling randomness (0.0 for deterministic DDIM).
        :return: The image tensor at the previous timestep.
        """
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        mapped_t = map_tensor[t]

        if self.CFG == 1.0:
            pred_noise = model(x, mapped_t, condition)
        else:
            unconditional_condition = self.unconditional_condition.unsqueeze(0).repeat(
                *([x.shape[0]] + [1] * len(self.unconditional_condition.shape)))
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([mapped_t] * 2)
            c_in = torch.cat([unconditional_condition, condition])
            noise_uncond, noise = model(x_in, t_in, c_in).chunk(2)
            pred_noise = noise_uncond + self.CFG * (noise - noise_uncond)

        # Extract the cumulative products for the current timestep
        alpha_cumprod_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_t_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)

        # Predict x0 from the current x and predicted noise
        pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)

        # Compute sigma_t for the DDIM step
        sigmas_t = (
                ddim_eta
                * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
                * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        # Compute the direction pointing to x_t
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

        # Generate additional noise for the step
        step_noise, _ = self.get_deterministic_noise_tensor(x.shape[0], x.shape[3])

        # Compute the image for the previous timestep
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * step_noise

        return x_prev

    def p_sample(self, model, x, t, condition=None, sampler="ddim"):
        """
        Sample one step backward in the diffusion process using the specified sampler.

        :param model: The neural network model for predicting noise.
        :param x: The current image tensor.
        :param t: The current timestep.
        :param condition: Optional conditioning information.
        :param sampler: The sampling method to use ('ddim' or 'ddpm').
        :return: The image tensor at the previous timestep.
        """
        if sampler == "ddim":
            return self.ddim_sample(model, x, t, condition=condition, ddim_eta=0.0)
        elif sampler == "ddpm":
            return self.ddim_sample(model, x, t, condition=condition, ddim_eta=1.0)
        else:
            raise NotImplementedError()

    def get_dynamic_masks(self, n_masks, shape, concat_points, mask_flexivity=0.8):
        """
        Generate dynamic masks for inpainting or guided sampling.

        The masks determine which parts of the image are frozen and which parts are updated.
        The last part (release area) is always kept.

        :param n_masks: Total number of masks to generate.
        :param shape: The shape of the image tensor.
        :param concat_points: List of indices where noise segments were concatenated.
        :param mask_flexivity: Ratio controlling the mask shrinkage over guidance steps.
        :return: A list of masks (each a tensor) for each diffusion step.
        """
        release_length = int(self.train_width / 4)
        assert shape[3] == (concat_points[-1] + release_length), "shape[3] != (concat_points[-1] + release_length)"

        # Compute the lengths of each segment between concatenation points
        fraction_lengths = [concat_points[i + 1] - concat_points[i] for i in range(len(concat_points) - 1)]

        # Determine how many steps will use guidance vs. free steps
        n_guidance_steps = int(n_masks * mask_flexivity)
        n_free_steps = n_masks - n_guidance_steps

        masks = []
        # For guidance steps, gradually shrink the mask to freeze parts of the image.
        for i in range(n_guidance_steps):
            # Initialize mask with zeros (0 means update, 1 means freeze)
            step_i_mask = torch.zeros((shape[0], 1, shape[2], shape[3]), dtype=torch.float32).to(self.device)
            # Always keep the release area at the end
            step_i_mask[:, :, :, -release_length:] = 1.0

            for fraction_index in range(len(fraction_lengths)):
                # Compute the length of the mask for this fraction in the current step

                _fraction_mask_length = int((n_guidance_steps - 1 - i) / (n_guidance_steps - 1) * fraction_lengths[fraction_index])

                if fraction_index == 0:
                    step_i_mask[:, :, :, :_fraction_mask_length] = 1.0
                elif fraction_index == len(fraction_lengths) - 1:
                    if not _fraction_mask_length == 0:
                        step_i_mask[:, :, :, -_fraction_mask_length - release_length:] = 1.0
                else:
                    # Center the mask in the fraction segment
                    fraction_mask_start_position = int((fraction_lengths[fraction_index] - _fraction_mask_length) / 2)

                    step_i_mask[:, :, :,
                    concat_points[fraction_index] + fraction_mask_start_position:concat_points[
                                                                                     fraction_index] + fraction_mask_start_position + _fraction_mask_length] = 1.0
            masks.append(step_i_mask)

        # For free steps, only the release area is kept.
        for i in range(n_free_steps):
            step_i_mask = torch.zeros((shape[0], 1, shape[2], shape[3]), dtype=torch.float32).to(self.device)
            step_i_mask[:, :, :, -release_length:] = 1.0
            masks.append(step_i_mask)

        masks.reverse()  # Reverse the list so that the masks are applied in the correct order.
        return masks

    @torch.no_grad()
    def p_sample_loop(self, model, shape, initial_noise=None, start_noise_level_ratio=1.0, end_noise_level_ratio=0.0,
                      return_tensor=False, condition=None, guide_img=None,
                      mask=None, sampler="ddim", inpaint=False, use_dynamic_mask=False, mask_flexivity=0.8):
        """
        Perform a full sampling loop from noise to the final image.

        Depending on the parameters, this function can perform pure sampling,
        image-guided sampling, or inpainting.

        :param model: The neural network model for noise prediction.
        :param shape: Shape of the image tensor [batch, channels, height, width].
        :param initial_noise: Optional starting noise tensor.
        :param start_noise_level_ratio: Ratio indicating the starting noise level.
        :param end_noise_level_ratio: Ratio indicating the ending noise level.
        :param return_tensor: If True, return torch tensors; otherwise, return numpy arrays.
        :param condition: Optional condition for guided sampling.
        :param guide_img: Optional guide image for image-guided sampling.
        :param mask: Optional mask for inpainting.
        :param sampler: Sampling method ('ddim' or 'ddpm').
        :param inpaint: If True, perform inpainting.
        :param use_dynamic_mask: If True, generate dynamic masks instead of using a fixed mask.
        :param mask_flexivity: Parameter controlling dynamic mask shrinkage.
        :return: A list of intermediate images and the initial noise tensor.
        """

        assert shape[1] == self.channels, "shape[1] != self.channels"
        assert shape[2] == self.height, "shape[2] != self.height"

        # Generate initial noise tensor if not provided
        initial_noise, _ = self.get_deterministic_noise_tensor(shape[0], shape[3], reference_noise=initial_noise)
        assert initial_noise.shape == shape, "initial_noise.shape != shape"

        # Calculate the start and end indices based on noise level ratios
        start_noise_level_index = int(self.num_timesteps * start_noise_level_ratio) # not included!!!
        end_noise_level_index = int(self.num_timesteps * end_noise_level_ratio)

        # Create a reversed range of timesteps for the sampling process
        timesteps = reversed(range(end_noise_level_index, start_noise_level_index))

        # If no guide image is provided, start from pure noise; otherwise, prepare the guide image.
        assert (start_noise_level_ratio == 1.0) or (
            not guide_img is None), "A guide_img must be given to sample from a non-pure-noise."

        if guide_img is None:
            img = initial_noise
        else:
            guide_img, concat_points = self.get_deterministic_noise_tensor_repeat(shape[0], shape[3], reference_noise=guide_img)
            assert guide_img.shape == shape, "guide_img.shape != shape"

            if start_noise_level_index > 0:
                t = torch.full((shape[0],), start_noise_level_index-1, device=self.device).long()   # -1 for start_noise_level_index not included
                img = self.q_sample(guide_img, t, noise=initial_noise)
            else:
                print("Zero noise added to the guidance latent representation.")
                img = guide_img

        # Prepare masks for inpainting/guided steps if needed.
        n_masks = start_noise_level_index - end_noise_level_index
        if use_dynamic_mask:
            masks = self.get_dynamic_masks(n_masks, shape, concat_points, mask_flexivity)
        else:
            masks = [mask for _ in range(n_masks)]

        imgs = [img]
        current_mask = None

        # Iterate over timesteps to progressively refine the image.
        for i in tqdm(timesteps, total=start_noise_level_index - end_noise_level_index, disable=self.mute):

            img = self.p_sample(model, img, torch.full((shape[0],), i, device=self.device, dtype=torch.long),
                                condition=condition,
                                sampler=sampler)

            # In inpainting mode, blend the generated image with the guided image using the mask.
            if inpaint:
                if i > 0:
                    t = torch.full((shape[0],), int(i-1), device=self.device).long()
                    img_noise_t = self.q_sample(guide_img, t, noise=initial_noise)
                    # if i == 3:
                    #     return [img_noise_t], initial_noise  # 第2排，第2列
                    current_mask = masks.pop()
                    img = current_mask * img_noise_t + (1 - current_mask) * img
                    # if i == 3:
                    #     return [img], initial_noise  # 第1.5排，最后1列
                else:
                    img = current_mask * guide_img + (1 - current_mask) * img

            if return_tensor:
                imgs.append(img)
            else:
                imgs.append(img.cpu().numpy())

        return imgs, initial_noise


    def sample(self, model, shape, return_tensor=False, condition=None, sampler="ddim", initial_noise=None, seed=None):
        """
        Generate a sample image from pure noise.

        :param model: The neural network model for noise prediction.
        :param shape: Desired shape of the output image tensor.
        :param return_tensor: If True, returns torch tensors; otherwise, returns numpy arrays.
        :param condition: Optional conditioning information.
        :param sampler: Sampling method ('ddim' or 'ddpm').
        :param initial_noise: Optional starting noise tensor.
        :param seed: Optional random seed for reproducibility.
        :return: The generated image(s) and the initial noise tensor.
        """
        if not seed is None:
            torch.manual_seed(seed)
        return self.p_sample_loop(model, shape, initial_noise=initial_noise, start_noise_level_ratio=1.0, end_noise_level_ratio=0.0,
                                  return_tensor=return_tensor, condition=condition, sampler=sampler)

    def interpolate(self, model, shape, variance, first_endpoint=None, second_endpoint=None, return_tensor=False,
                    condition=None, sampler="ddim", seed=None):
        """
        Generate an interpolated image by linearly interpolating noise between two endpoints.

        :param model: The neural network model for noise prediction.
        :param shape: Desired shape of the output image tensor.
        :param variance: Target variance for the generated noise.
        :param first_endpoint: Optional first endpoint tensor.
        :param second_endpoint: Optional second endpoint tensor.
        :param return_tensor: If True, returns torch tensors; otherwise, returns numpy arrays.
        :param condition: Optional conditioning information.
        :param sampler: Sampling method.
        :param seed: Optional random seed.
        :return: The interpolated image(s) and the initial noise tensor.
        """
        if not seed is None:
            torch.manual_seed(seed)
        linear_noise = self.generate_linear_noise(shape, variance, first_endpoint=first_endpoint,
                                                  second_endpoint=second_endpoint)
        return self.p_sample_loop(model, shape, initial_noise=linear_noise, start_noise_level_ratio=1.0,
                                  end_noise_level_ratio=0.0,
                                  return_tensor=return_tensor, condition=condition, sampler=sampler)

    def img_guided_sample(self, model, shape, noising_strength, guide_img, return_tensor=False, condition=None,
                          sampler="ddim", initial_noise=None, seed=None):
        """
        Perform style-transfer.

        :param model: The neural network model for noise prediction.
        :param shape: Desired shape of the output image tensor.
        :param noising_strength: The level of noise to add (as a ratio) to the guide image.
        :param guide_img: The guide image tensor.
        :param return_tensor: If True, return torch tensors; otherwise, return numpy arrays.
        :param condition: Optional conditioning information.
        :param sampler: Sampling method.
        :param initial_noise: Optional starting noise tensor.
        :param seed: Optional random seed.
        :return: The generated image(s) and the initial noise tensor.
        """
        if not seed is None:
            torch.manual_seed(seed)
        assert guide_img.shape[-1] == shape[-1], "guide_img.shape[:-1] != shape[:-1]"
        return self.p_sample_loop(model, shape, start_noise_level_ratio=noising_strength, end_noise_level_ratio=0.0,
                                  return_tensor=return_tensor, condition=condition, sampler=sampler,
                                  guide_img=guide_img, initial_noise=initial_noise)

    def inpaint_sample(self, model, shape, noising_strength, guide_img, mask, return_tensor=False, condition=None,
                       sampler="ddim", initial_noise=None, use_dynamic_mask=False, end_noise_level_ratio=0.0, seed=None,
                       mask_flexivity=0.8):
        """
        Perform inpainting by sampling with a given guide image and mask.

        :param model: The neural network model for noise prediction.
        :param shape: Desired shape of the output image tensor.
        :param noising_strength: The starting noise level ratio.
        :param guide_img: The guide image tensor to inpaint from.
        :param mask: A fixed mask tensor indicating the region to keep.
        :param return_tensor: If True, return torch tensors; otherwise, return numpy arrays.
        :param condition: Optional conditioning information.
        :param sampler: Sampling method.
        :param initial_noise: Optional starting noise tensor.
        :param use_dynamic_mask: Whether to generate dynamic masks.
        :param end_noise_level_ratio: Ending noise level ratio.
        :param seed: Optional random seed.
        :param mask_flexivity: Parameter controlling dynamic mask behavior.
        :return: The inpainted image(s) and the initial noise tensor.
        """
        if not seed is None:
            torch.manual_seed(seed)
        return self.p_sample_loop(model, shape, start_noise_level_ratio=noising_strength, end_noise_level_ratio=end_noise_level_ratio,
                                  return_tensor=return_tensor, condition=condition, guide_img=guide_img, mask=mask,
                                  sampler=sampler, inpaint=True, initial_noise=initial_noise, use_dynamic_mask=use_dynamic_mask,
                                  mask_flexivity=mask_flexivity)