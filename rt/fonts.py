"""Download the Hebrew font set (all OFL, Google Fonts) and write a catalog with style tags."""
import os, json, urllib.request
from fontTools.ttLib import TTFont

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FONTS = {
    # name: (path, tags)
    "Heebo": ("heebo/Heebo%5Bwght%5D.ttf", "sans, clean, modern, variable weight 100-900 - default for plain titles"),
    "Assistant": ("assistant/Assistant%5Bwght%5D.ttf", "sans, humanist, light-to-extrabold"),
    "Rubik": ("rubik/Rubik%5Bwght%5D.ttf", "sans, slightly rounded corners, friendly, variable 300-900"),
    "Varela Round": ("varelaround/VarelaRound-Regular.ttf", "rounded sans, soft"),
    "Fredoka": ("fredoka/Fredoka%5Bwdth,wght%5D.ttf", "rounded, chunky, playful - kids/animation titles"),
    "Secular One": ("secularone/SecularOne-Regular.ttf", "heavy display sans, poster/action"),
    "Suez One": ("suezone/SuezOne-Regular.ttf", "heavy slab-serif display, epic/classic"),
    "Frank Ruhl Libre": ("frankruhllibre/FrankRuhlLibre%5Bwght%5D.ttf", "classic serif, book/drama, variable"),
    "David Libre": ("davidlibre/DavidLibre-Regular.ttf", "classic serif"),
    "David Libre Bold": ("davidlibre/DavidLibre-Bold.ttf", "classic serif bold"),
    "Bellefair": ("bellefair/Bellefair-Regular.ttf", "elegant thin serif, luxury/romance/trajan-like"),
    "Miriam Libre": ("miriamlibre/MiriamLibre%5Bwght%5D.ttf", "geometric sans, technical"),
    "Alef": ("alef/Alef-Regular.ttf", "neutral sans"),
    "Alef Bold": ("alef/Alef-Bold.ttf", "neutral sans bold"),
    "Karantina": ("karantina/Karantina-Regular.ttf", "condensed display, tall narrow - thriller/horror"),
    "Karantina Bold": ("karantina/Karantina-Bold.ttf", "condensed display bold"),
    "Karantina Light": ("karantina/Karantina-Light.ttf", "condensed display light"),
    "Amatic SC": ("amaticsc/AmaticSC-Bold.ttf", "hand-drawn narrow marker"),
    "Gveret Levin": ("gveretlevin/GveretLevin-Regular.ttf", "handwritten brush/marker, casual"),
    "Solitreo": ("solitreo/Solitreo-Regular.ttf", "historic cursive"),
    "Noto Sans Hebrew": ("notosanshebrew/NotoSansHebrew%5Bwdth,wght%5D.ttf", "neutral sans, widths+weights"),
    "Noto Serif Hebrew": ("notoserifhebrew/NotoSerifHebrew%5Bwdth,wght%5D.ttf", "neutral serif, widths+weights"),
    "IBM Plex Sans Hebrew Bold": ("ibmplexsanshebrew/IBMPlexSansHebrew-Bold.ttf", "corporate sans bold"),
    "Rubik Mono One": ("rubikmonoone/RubikMonoOne-Regular.ttf", "ultra-heavy wide, games/arcade"),
    "Rubik Dirt": ("rubikdirt/RubikDirt-Regular.ttf", "heavy, grunge texture"),
    "Rubik Glitch": ("rubikglitch/RubikGlitch-Regular.ttf", "glitch/digital"),
    "Rubik Wet Paint": ("rubikwetpaint/RubikWetPaint-Regular.ttf", "dripping, horror/halloween"),
    "Rubik Spray Paint": ("rubikspraypaint/RubikSprayPaint-Regular.ttf", "graffiti spray"),
    "Rubik Bubbles": ("rubikbubbles/RubikBubbles-Regular.ttf", "bubble, kids"),
    "Rubik Burned": ("rubikburned/RubikBurned-Regular.ttf", "burned/fire"),
    "Rubik Distressed": ("rubikdistressed/RubikDistressed-Regular.ttf", "worn/distressed"),
    "Rubik Moonrocks": ("rubikmoonrocks/RubikMoonrocks-Regular.ttf", "rocky/space"),
    "Rubik Iso": ("rubikiso/RubikIso-Regular.ttf", "isometric 3D"),
    "Rubik Vinyl": ("rubikvinyl/RubikVinyl-Regular.ttf", "retro striped"),
    "Rubik 80s Fade": ("rubik80sfade/Rubik80sFade-Regular.ttf", "80s retro fade lines"),
    "Rubik Beastly": ("rubikbeastly/RubikBeastly-Regular.ttf", "monster/hairy"),
    "Rubik Gemstones": ("rubikgemstones/RubikGemstones-Regular.ttf", "faceted gem"),
    "Rubik Marker Hatch": ("rubikmarkerhatch/RubikMarkerHatch-Regular.ttf", "hatched marker"),
    "Rubik Scribble": ("rubikscribble/RubikScribble-Regular.ttf", "scribble fill"),
    "Rubik Maze": ("rubikmaze/RubikMaze-Regular.ttf", "maze pattern"),
    "Rubik Microbe": ("rubikmicrobe/RubikMicrobe-Regular.ttf", "organic blobs"),
    "Rubik Puddles": ("rubikpuddles/RubikPuddles-Regular.ttf", "puddle/liquid"),
    "Rubik Broken Fax": ("rubikbrokenfax/RubikBrokenFax-Regular.ttf", "broken fax/print"),
}


def run(dest):
    os.makedirs(dest, exist_ok=True)
    cat = {}
    for name, (path, tags) in FONTS.items():
        fn = path.split("/")[-1].replace("%5B", "[").replace("%5D", "]")
        p = os.path.join(dest, fn)
        if not os.path.exists(p):
            try:
                urllib.request.urlretrieve(BASE + path, p)
            except Exception as e:
                print("skip", name, e); continue
        try:
            cmap = TTFont(p).getBestCmap()
        except Exception:
            continue
        if 0x05D0 not in cmap:  # no Hebrew alef
            os.remove(p); print("no hebrew:", name); continue
        cat[name] = {"file": fn, "tags": tags}
    with open(os.path.join(dest, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=1)
    print(len(cat), "fonts")
    return cat
