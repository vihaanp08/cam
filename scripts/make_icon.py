from PIL import Image, ImageDraw, ImageFilter
S=1024
img=Image.new("RGB",(S,S))
d=ImageDraw.Draw(img)
for y in range(S):
    t=y/S
    c=int(58-30*t)
    d.line([(0,y),(S,y)],fill=(c,c,c+4))
cx,cy=S//2,S//2+10
def circ(r,fill=None,outline=None,w=0):
    d.ellipse([cx-r,cy-r,cx+r,cy+r],fill=fill,outline=outline,width=w)
circ(330,fill=(196,198,204))
circ(300,fill=(28,29,34))
circ(232,fill=(52,56,70))
circ(170,fill=(14,16,24))
circ(104,fill=(36,44,74))
circ(46,fill=(8,9,14))
hl=Image.new("L",(S,S),0)
hd=ImageDraw.Draw(hl)
hd.ellipse([cx-120,cy-150,cx-40,cy-70],fill=170)
hl=hl.filter(ImageFilter.GaussianBlur(14))
img.paste((235,238,255),(0,0),hl)
d.ellipse([S-250,150,S-190,210],fill=(255,196,60))
img.save("Cam/Assets.xcassets/AppIcon.appiconset/icon-1024.png")
