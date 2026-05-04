import FastGeodis
import torch


def fast_gdf_wrapper(
    image: torch.Tensor,
    start_x: int,
    start_y: int,
    obstacle_gdf_value: float = 1000.0,
    iterations: int = 2,
) -> torch.Tensor:
    """Geodesic distance field from a binary obstacle image.

    Args:
        image: (1, 1, H, W) float32 - 0 = free, >0 = obstacle (scaled internally).
        start_x, start_y: goal cell indices.
        obstacle_gdf_value: value assigned to unreachable cells.
    Returns:
        gdf: (1, 1, H, W) geodesic distances in the same units as the resolution caller
             uses (caller must multiply by resolution to get metres).
    """
    mult = image.shape[-1] * image.shape[-2]
    image = image.clone() * mult

    mask = torch.ones_like(image)
    mask[..., start_x, start_y] = 0

    image[..., 0, :] = 0
    image[..., -1, :] = 0
    image[..., :, 0] = 0
    image[..., :, -1] = 0

    v = 1e10
    lamb = 0.5
    gdf = FastGeodis.generalised_geodesic2d(image, mask, v, lamb, iterations)
    gdf *= 2

    if gdf[..., image.shape[-2] // 2, image.shape[-1] // 2] > mult:
        gdf = FastGeodis.generalised_geodesic2d(image, mask, v, 0.0, iterations)
        gdf *= 2

    gdf[gdf > mult] = obstacle_gdf_value
    return gdf
