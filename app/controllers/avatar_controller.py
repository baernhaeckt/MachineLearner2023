from fastapi import Body, HTTPException, Request, APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from models.configure_avatar import ConfigureAvatar

import config
import python_avatars as pa
import cairosvg
from PIL import Image, ImageOps, ImageChops, ImageFilter
from collections import deque
import io
import replicate
import os
import requests
import base64
import random
import math

router = APIRouter()

cache_folder = os.environ["AVATAR_IMAGECACHE_FOLDER"] if "AVATAR_IMAGECACHE_FOLDER" in os.environ and not os.environ["AVATAR_IMAGECACHE_FOLDER"] is None else "cache"
if not cache_folder.endswith("/"):
    cache_folder += "/"

if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

def get_enum_name_by_value(enum_class, value):
    for name, member_value in vars(enum_class).items():
        if member_value == value:
            return name
    return None

def generate_avatar(input_hairtype=None, input_haircolor=None, input_skincolor=None, input_facialhairtype=None, input_clothing=None):

    valid_hairtype = [
        pa.HairType.BIG_HAIR,
        pa.HairType.BOB,
        pa.HairType.CURLY,
        pa.HairType.CURVY,
        pa.HairType.FRIZZLE,
        pa.HairType.FRO,
        pa.HairType.BRAIDS,
        pa.HairType.BUZZCUT,
        pa.HairType.DREADLOCKS,
        pa.HairType.EINSTEIN_HAIR,
    ]

    hair_type = pa.HairType(input_hairtype) if not input_hairtype is None else random.choice(valid_hairtype)
    hair_color = pa.HairColor(input_haircolor) if not input_haircolor is None else random.choice([pa.HairColor.BLONDE, pa.HairColor.BLACK])
    skin_color = pa.SkinColor(input_skincolor) if not input_skincolor is None else random.choice([pa.SkinColor.LIGHT, pa.SkinColor.DARK_BROWN])
    facial_hair = pa.FacialHairType(input_facialhairtype) if not input_facialhairtype is None else random.choice([pa.FacialHairType.NONE, pa.FacialHairType.NONE, pa.FacialHairType.BEARD_LIGHT, pa.FacialHairType.BEARD_MAGESTIC]) # 2/3 chance of no facial hair
    clothing = pa.ClothingType(input_clothing) if not input_clothing is None else random.choice([pa.ClothingType.HOODIE, pa.ClothingType.SHIRT_V_NECK, pa.ClothingType.COLLAR_SWEATER, pa.ClothingType.SHIRT_CREW_NECK])

    avatar = pa.Avatar(
        style=pa.AvatarStyle.TRANSPARENT,
        background_color=pa.BackgroundColor.WHITE,
        eyebrows=pa.EyebrowType.DEFAULT,
        eyes=pa.EyeType.DEFAULT,
        nose=pa.NoseType.pick_random(),
        mouth=pa.MouthType.DEFAULT,
        accessory=pa.AccessoryType.pick_random(),
        clothing_color=pa.ClothingColor.pick_random(),
        clothing=clothing,
        top=hair_type,
        facial_hair=facial_hair,
        skin_color=skin_color,
        hair_color=hair_color,
    )

    return avatar

def convert_png_to_jpg(png_byte_data):

    # Convert PNG bytes to PIL Image
    png_image = Image.open(io.BytesIO(png_byte_data))

    # Create a new white background image
    white_background = Image.new("RGBA", png_image.size, "WHITE") 
    
    # Paste the PNG (with alpha) onto the white background
    white_background.paste(png_image, (0, 0), png_image)
    
    # Convert the resulting RGBA image to RGB (discarding alpha)
    final_image = white_background.convert("RGB")

    # Convert to JPG and save in-memory
    jpg_byte_stream = io.BytesIO()
    final_image.save(jpg_byte_stream, format="JPEG")
    
    # Get the JPG byte data
    return jpg_byte_stream.getvalue()

def convert_all_svg_to_jpg(svg_data_array):

    result = []
    for svg_data in svg_data_array:
        # Convert SVG to PNG in-memory
        png_byte_data = cairosvg.svg2png(bytestring=svg_data[1], output_width=512, output_height=512)

        result.append([svg_data[0], convert_png_to_jpg(png_byte_data)])
    
    return result

def download_file(url):
    # Download the image
    response = requests.get(url)

    # Check if the download was successful
    if response.status_code == 200:
        # Get byte data
        byte_data = response.content

        return byte_data
    else:
        print(f"Failed to download image. HTTP Status Code: {response.status_code}")

def image_bytes_to_data_uri(image_bytes, image_format="png"):
    """
    Convert image bytes to a Data URI.
    
    Parameters:
    - image_bytes (bytes): The byte array of the image.
    - image_format (str): The format of the image (e.g., "png", "jpg").

    Returns:
    str: The Data URI string.
    """
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format};base64,{base64_str}"

def color_diff(color1, color2):
    """Calculate the Euclidean distance between two colors in RGB space."""
    return math.sqrt((color1[0] - color2[0]) ** 2 + (color1[1] - color2[1]) ** 2 + (color1[2] - color2[2]) ** 2)

def flood_fill(image, x, y, target_color, fill_color, threshold=10):
    """Perform flood fill algorithm with thresholding."""
    width, height = image.size
    pixels = image.load()
    
    if color_diff(pixels[x, y], target_color) > threshold:
        return
    
    q = deque([(x, y)])

    while q:
        x, y = q.popleft()

        if color_diff(pixels[x, y], target_color) <= threshold:
            pixels[x, y] = fill_color
            if x > 0:
                q.append((x-1, y))
            if x < width - 1:
                q.append((x+1, y))
            if y > 0:
                q.append((x, y-1))
            if y < height - 1:
                q.append((x, y+1))

def create_image_mask(input_image_byte_data):
    # Open the original image
    image = Image.open(io.BytesIO(input_image_byte_data)).convert("RGB")

    image.save(f'{cache_folder}prompt_image.jpg', format="JPEG")

    # Step 1: Invert the colors
    inverted_image = ImageOps.invert(image)

    # Step 2: Perform flood fill from the borders with a "not-used" color
    width, height = inverted_image.size
    target_color = (0, 0, 0)  # black in inverted image
    not_used_color = (128, 128, 128)  # a "not-used" color

    for x in range(width):
        flood_fill(inverted_image, x, 0, target_color, not_used_color)
        flood_fill(inverted_image, x, height-1, target_color, not_used_color)

    for y in range(height):
        flood_fill(inverted_image, 0, y, target_color, not_used_color)
        flood_fill(inverted_image, width-1, y, target_color, not_used_color)

    # Step 3: Change each non-"not-used" pixel to white and each "not-used" pixel to black
    pixels = inverted_image.load()
    for x in range(width):
        for y in range(height):
            if pixels[x, y] != not_used_color:
                pixels[x, y] = (255, 255, 255)  # white
            else:
                pixels[x, y] = (0, 0, 0)  # black

    # Convert to JPG and save in-memory
    jpg_byte_stream = io.BytesIO()
    inverted_image.save(jpg_byte_stream, format="JPEG")
    inverted_image.save(f'{cache_folder}/mask.jpg', format="JPEG")
    
    # Get the JPG byte data
    return jpg_byte_stream.getvalue()

def generate_avatar_photos(jpg_byte_data_array):

    if os.environ.get("REPLICATE_API_TOKEN") is None or os.environ.get("REPLICATE_API_TOKEN") == "":
        raise Exception("REPLICATE_API_TOKEN environment variable not set")
    
    result = []
    seed = random.randint(1000, 100000)

    for input_data in jpg_byte_data_array:
        input_image = input_data[1]
        features = input_data[0]
        prompt = f"Front portrait photography of a real {features[0]} with a {', '.join(features[1:-1])} against a white background. Daylight lightning, natural skin coloring. 50mm portrait lens photography, award winning. Without text or watermark. Shot by a professional photographer."
        model_input = {
            "prompt": prompt,
            # "negative_prompt": "cartoon style, comic style, svg graphic, svg icon, stylized, Painting, drawing, handdrawn, Text, dirty teeth, empty eyes, creepy, child",
            "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, dotted:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, cartoon style, comic style, svg graphic, svg icon, stylized, Painting, drawing, handdrawn, Text, dirty teeth, empty eyes, creepy, child",
            "width": 512,
            "height": 512,
            "image": image_bytes_to_data_uri(input_image),
            "mask": image_bytes_to_data_uri(create_image_mask(input_image)),
            "prompt_strength": 0.8,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "scheduler": "DPMSolverMultistep",
            "seed": seed
        }

        print(prompt)

        try:
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:c11bac58203367db93a3c552bd49a25a5418458ddffb7e90dae55780765e26d6",
                input=model_input
            )
            
            output_url = output[0]
            ouptut_png_data = download_file(output_url)
            result.append(ouptut_png_data)
        except Exception as e:
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:c11bac58203367db93a3c552bd49a25a5418458ddffb7e90dae55780765e26d6",
                input=model_input
            )
            
            output_url = output[0]
            ouptut_png_data = download_file(output_url)
            result.append(ouptut_png_data)

    return result

def calculate_cache_filename(avatar):
    return f"avatar_{avatar.clothing}_{avatar.top}_{avatar.facial_hair}_{avatar.skin_color}_{avatar.hair_color}.png".replace("#", "").lower()

def calculate_emotion_cache_filename(cache_filename, emotion):
    return f"{cache_filename}".replace(".png", f"_{emotion}.png")

def save_to_cache(cache_filename, avatar_photo, emotion):
    target_filename = calculate_emotion_cache_filename(cache_filename, emotion)
    with open(f"{cache_folder}{target_filename}", "wb") as f:
        f.write(avatar_photo)

    return target_filename

def extract_features(avatar: pa.Avatar, facial_expression):
    male_features = [
        pa.HairType.NONE,
        pa.HairType.BUZZCUT,
        pa.HairType.CAESAR,
        pa.HairType.EINSTEIN_HAIR,
        pa.FacialHairType.BEARD_LIGHT,
        pa.FacialHairType.BEARD_MAGESTIC,
    ]

    is_female = True
    if avatar.top in male_features or avatar.facial_hair in male_features:
        is_female = False

    extracted_features = []
    
    extracted_features.append("Cool woman" if is_female else "Cool man")
    extracted_features.append(f"{get_enum_name_by_value(pa.SkinColor, avatar.skin_color)} skin tone".replace("_", " ").lower())
    extracted_features.append(f"wearing a {avatar.clothing}".replace("_", " ").lower())
    
    if (avatar.top != pa.HairType.NONE):
        hair_color_name = get_enum_name_by_value(pa.HairColor, avatar.hair_color)
        extracted_features.append(f"with {hair_color_name} {avatar.top} hair".replace("_", " ").lower())
    else:
        extract_features.append("with a bald head".replace("_", " ").lower())

    extracted_features.append(f"{avatar.facial_hair} as facial hair".replace("_", " ").lower() if avatar.facial_hair != pa.FacialHairType.NONE else "without facial hair")
    extracted_features.append(f"with a {avatar.accessory} as accessory".replace("_", " ").lower() if avatar.accessory != pa.AccessoryType.NONE else "without an accessory")
    extracted_features.append(f"with a {facial_expression} facial expression")

    return extracted_features

def initialize_avatar(avatar: pa.Avatar, request: Request):
    avatar_base_filename = calculate_cache_filename(avatar)
    base_url = str(request.base_url) + "api/v1/avatar/"

    avatar_configuration = {
        "hairId": avatar.top,
        "hairColorId": avatar.hair_color,
        "skinColorId": avatar.skin_color,
        "facialHairId": avatar.facial_hair,
        "clothingId": avatar.clothing,
    }

    if os.path.exists(f"{cache_folder}{calculate_emotion_cache_filename(avatar_base_filename, 'default')}"):
        return {
            "default": base_url + calculate_emotion_cache_filename(avatar_base_filename, "default"),
            "configuration": avatar_configuration
            # "sad": base_url + calculate_emotion_cache_filename(avatar_base_filename, "sad"),
            # "happy": base_url + calculate_emotion_cache_filename(avatar_base_filename, "happy"),
            # "frightened": base_url + calculate_emotion_cache_filename(avatar_base_filename, "frightened"),
            # "confused": base_url + calculate_emotion_cache_filename(avatar_base_filename, "confused"),
        }

    default = avatar
    # sad_avatar = avatar.sad()
    # happy_avatar = avatar.happy()
    # frightened_avatar = avatar.frightened()
    # confused_avatar = avatar.confused()

    svg_data_array = [
        [extract_features(default, "neutral"), default.render()],
        # [extract_features(sad_avatar, "sad"), sad_avatar.render()],
        # [extract_features(happy_avatar, "happy"), happy_avatar.render()],
        # [extract_features(frightened_avatar, "frightened"), frightened_avatar.render()],
        # [extract_features(confused_avatar, "confused"), confused_avatar.render()],
    ]

    jpg_byte_data_array = convert_all_svg_to_jpg(svg_data_array)
    avatar_photo_array = generate_avatar_photos(jpg_byte_data_array)
    
    result = {
        "default": base_url + save_to_cache(avatar_base_filename, avatar_photo_array[0], "default"),
        "configuration": avatar_configuration
        # "sad": base_url + save_to_cache(avatar_base_filename, avatar_photo_array[1], "sad"),
        # "happy": base_url + save_to_cache(avatar_base_filename, avatar_photo_array[2], "happy"),
        # "frightened": base_url + save_to_cache(avatar_base_filename, avatar_photo_array[3], "frightened"),
        # "confused": base_url + save_to_cache(avatar_base_filename, avatar_photo_array[4], "confused"),
    }
    
    return result

@router.post("/configure", tags=["api avatar"], status_code=200)
def configure_avatar(request: Request, model: ConfigureAvatar = Body(...)):
    avatar = generate_avatar(model.hairId, model.hairColorId, model.skinColorId, model.facialHairId, model.clothingId)

    return initialize_avatar(avatar, request)

@router.get("/{cache_filename}", tags=["api avatar"], status_code=200)
def get_avatar_from_cache(cache_filename):
    if not os.path.exists(f"{cache_folder}{cache_filename}"):
        return {"message": "Avatar not found"}, 404

    if os.path.exists(f"{cache_folder}{cache_filename}"):
        return FileResponse(f"{cache_folder}{cache_filename}")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@router.get("/", tags=["api avatar"], status_code=200)
def get_avatar(request: Request):

    avatar = generate_avatar()

    return initialize_avatar(avatar, request)
