from PIL import Image
import sys

def main():
    fname = sys.argv[1]
    image = Image.open(fname)
    rgb_im = image.convert('RGB')
    width, height = rgb_im.size
    image_new = Image.new('RGB', (width, height))
    pixelsNew = image_new.load()
    for w in range(0, width):
        for h in range(0, height):
            r, g, b = rgb_im.getpixel((w, h))
            pixelsNew[w,h] = (int(r/2), int(g/2), int(b/2) )
    image.close()
    image_new.save("Q2.jpg")
    image_new.close()
    #print(r, g, b)

    

if __name__ == "__main__":
    main()
