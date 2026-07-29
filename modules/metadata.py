from PIL import Image
from PIL.ExifTags import TAGS

class MetadataVerifier:

    def analyze(self, image_path):

        print("\n========== METADATA ANALYSIS ==========\n")

        try:
            image = Image.open(image_path)

            print("Image Size :", image.size)
            print("Image Format :", image.format)

            exif = image._getexif()

            if exif is None:
                print("\nNo EXIF Metadata Found")
                return

            exif_data = {}

            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value

            make = exif_data.get("Make", "Not Available")
            model = exif_data.get("Model", "Not Available")
            software = exif_data.get("Software", "Not Available")
            datetime = exif_data.get("DateTime", "Not Available")

            print("\nCamera Make :", make)
            print("Camera Model :", model)
            print("Software :", software)
            print("Date Taken :", datetime)

            score = 0

            if make != "Not Available":
                score += 30

            if model != "Not Available":
                score += 30

            if datetime != "Not Available":
                score += 20

            if software != "Not Available":
                score += 20

            print("\nMetadata Score :", score, "/100")

            if score >= 70:
                print("Status : Authentic Metadata")

            elif score >= 40:
                print("Status : Partially Available")

            else:
                print("Status : Suspicious")

        except Exception as e:
            print(e)