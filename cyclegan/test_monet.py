from cycle_gan import generate_image

input_image = "test_images/green_hills.jpg"
output_image = "output_images/monet.jpg"

# mode=1 for photo → Monet | mode=2 for Monet → photo
generate_image(input_image, output_image, mode=2)