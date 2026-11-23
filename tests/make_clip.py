import numpy as np, cv2, subprocess, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
FONT=os.path.join(HERE,'..','fonts','Heebo[wght].ttf')
OUT=sys.argv[1] if len(sys.argv)>1 else 'test.mp4'
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1280,720,24
rng=np.random.default_rng(1)
def tex(seed,col):
    r=np.random.default_rng(seed)
    big=np.zeros((1200,2400,3),np.float32)
    for k in range(400):
        x,y=r.integers(0,2400),r.integers(0,1200); rad=int(r.integers(10,90))
        c=np.array(col)*r.uniform(.4,1.2)+r.uniform(-40,40,3)
        cv2.circle(big,(int(x),int(y)),rad,c.tolist(),-1)
    big=cv2.GaussianBlur(big,(0,0),3)
    big+=r.normal(0,6,big.shape)
    return np.clip(big,0,255).astype(np.uint8)
A=tex(1,(200,120,60)); B=tex(2,(60,140,90))
F=ImageFont.truetype(FONT,84); F.set_variation_by_axes([800])
F2=ImageFont.truetype(FONT,56); F2.set_variation_by_axes([800])
def txt(img,s,font,cx,cy,fill,scale=1,alpha=1,shadow=None,outline=0):
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    b=d.textbbox((0,0),s,font=font,stroke_width=outline)
    x=cx-(b[2]+b[0])/2; y=cy-(b[3]+b[1])/2
    if shadow: d.text((x+shadow,y+shadow),s,font=font,fill=(20,10,0,255))
    d.text((x,y),s,font=font,fill=fill+(255,),stroke_width=outline,stroke_fill=(0,0,0,255))
    if scale!=1:
        M=cv2.getRotationMatrix2D((cx,cy),0,scale); a=np.array(L)
        L=Image.fromarray(cv2.warpAffine(a,M,(W,H)))
    a=np.array(L).astype(np.float32)/255; a[...,3]*=alpha
    rgb=img[...,::-1].astype(np.float32)/255
    o=a[...,:3]*a[...,3:]+rgb*(1-a[...,3:])
    return (o[...,::-1]*255).astype(np.uint8)
p=subprocess.Popen([os.environ.get('FFMPEG','ffmpeg'),'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-','-f','lavfi','-i','sine=f=440:d=6','-shortest','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','aac',OUT],stdin=subprocess.PIPE)
for f in range(144):
    if f<72:
        x=int(100+f*6); y=int(200+f*1.5); img=A[y:y+H,x:x+W].copy()
        if 24<=f<=60:
            k=f-24; sc=min(1,0.6+0.4*k/6)
            img=txt(img,"IT'S TIME",F,640,300,(255,150,20),scale=sc,shadow=6)
        if 30<=f<=66:
            al=min(1,(f-30+1)/8, (66-f+1)/8)
            img=txt(img,"FOR A TRUE HERO",F2,640,470,(255,255,255),alpha=al,outline=3)
    else:
        x=int(300+(f-72)*3); img=B[300:300+H,x:x+W].copy()
        if 80<=f<=130:
            img=txt(img,"FROM THE STUDIO",F2,640,600,(240,240,240),outline=4)
    p.stdin.write(img.tobytes())
p.stdin.close(); p.wait()
