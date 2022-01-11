from PIL import Image
from PIL import ImageEnhance


class DataCentric:
    def flip_image(self, image_path, saved_location):
        """
        Flip or mirror the image

        @param image_path: The path to the image to edit
        @param saved_location: Path to save the cropped image
        """
        image_obj = Image.open(image_path)
        rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        rotated_image.save(saved_location)
        rotated_image.show()

    def adjust_sharpness(self, input_image, output_image, factor):
        image = Image.open(input_image)
        enhancer_object = ImageEnhance.Sharpness(image)
        out = enhancer_object.enhance(factor)
        out.save(output_image)
