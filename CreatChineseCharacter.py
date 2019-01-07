# -*- coding:utf-8 -*-
import pygame
import numpy as np
import os
import cv2
import random
import math


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# read all the chinese words and the fonts
def readChineseGB1_and_fonts(word_filename,fonts_filename):
      f = open(word_filename, "r",encoding="utf-8")  # 读文件
      lines = f.readlines()
      fonts_file = open(fonts_filename, "r")  # 读文件
      fonts = fonts_file.readlines()
      return lines[0],fonts

# get one fonts
def random_fonts(fonts):
      ff = random.randint(0,len(fonts)-1)
      font = fonts[ff]
      font = font.strip('\n')
      return font

def pasteWord(words,name,fontdir):
      '''输入一个文字，输出一张包含该文字的图片'''
      pygame.init()
      font = pygame.font.Font(fontdir, 200)
      i = 0
      for word in words:
            # print(word)
            text = word#.decode('utf-8')
            # word = unichr(text)
            rtext = font.render(text, True, (0, 0, 0),(255, 255, 255))
            # print(rtext)

            # pygame.image.save(rtext, imgName)
            img1 = pygame.surfarray.pixels2d(rtext)
            img1 = 255 - np.transpose(img1)
            img1 = cv2.resize(img1,(200,200),interpolation=cv2.INTER_CUBIC)

            # rtext2 = font.render(text, True, (0, 0, 0), (255, 255, 255))
            # img2 = pygame.surfarray.pixels2d(rtext2)
            # img2 = 255 - np.transpose(img2)
            if(i ==0 ):
                  img = img1
            else:
                  img = np.hstack((img, img1))
            i=i+1
      rows,cols = img.shape
      # 图像旋转
      theta = random.randint(-15,15)
      angle = -theta * math.pi / 180
      a = math.sin(angle)
      b = math.cos(angle)
      width = int(cols * math.fabs(b) + rows * math.fabs(a))
      heigth = int(rows * math.fabs(b) + cols * math.fabs(a))
      M = cv2.getRotationMatrix2D((width/2,heigth/2),theta,1)
      rot_move = np.dot(M,np.array([(width-cols)*0.5,(heigth-rows)*0.5,0]))
      M[0,2] += rot_move[0]
      M[1, 2] += rot_move[1]
      imgout_xuanzhun = cv2.warpAffine(img,M,(width,heigth),2,0,1)
      #图像透视
      y1,x1 = imgout_xuanzhun.shape
      pts1 = np.float32([[0,0],[x1,0],[0,y1],[x1,y1]])

      x2_1 = random.randint(0, int(x1 / 4))
      y2_1 = random.randint(0, int(y1 / 4))

      x2_2 = random.randint(0, int(x1 / 4))
      y2_2 = random.randint(0, int(y1 / 4))

      x2_3 = random.randint(0, int(x1 / 4))
      y2_3 = random.randint(0, int(y1 / 4))

      x2_4 = random.randint(0, int(x1 / 4))
      y2_4 = random.randint(0, int(y1 / 4))

      # out_w = x1 - min(x2_1,x2_2)-min(x2_3,x2_4)
      # out_h = y1 - min(y2_1,y2_3)-min(y2_2,y2_4)

      pts2 = np.float32([[x2_1, y2_1], [x1-x2_2, y2_2],
                         [x2_3, y1-y2_3], [x1-x2_4, y1-y2_4]])
      MM = cv2.getPerspectiveTransform(pts1,pts2)
      imgout_p = cv2.warpPerspective(imgout_xuanzhun,MM,(x1,y1),2,0,1)
      imgout = imgout_p[min(y2_1,y2_2):y1 -min(y2_3,y2_4),min(x2_1,x2_3):x1-min(x2_2,x2_4)]
      imgout = cv2.resize(imgout,(256,256),interpolation=cv2.INTER_CUBIC)
      imgName = "./creatdata/" + name+ ".png"
      cv2.imencode('.png',imgout)[1].tofile(imgName)
      # cv2.imwrite(imgName, imgout)


##添加运动模糊
def motion_blur(image, degree, angle):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

#图像透视
def perspective_trans(img):
    y1,x1 = img.shape
    pts1 = np.float32([[0,0],[x1,0],[0,y1],[x1,y1]])

    x2_1 = random.randint(0, int(x1 / 8))
    y2_1 = random.randint(0, int(y1 / 8))

    x2_2 = random.randint(0, int(x1 / 8))
    y2_2 = random.randint(0, int(y1 / 8))

    x2_3 = random.randint(0, int(x1 / 8))
    y2_3 = random.randint(0, int(y1 / 8))

    x2_4 = random.randint(0, int(x1 / 8))
    y2_4 = random.randint(0, int(y1 / 8))

    pts2 = np.float32([[x2_1, y2_1], [x1-x2_2, y2_2],
                       [x2_3, y1-y2_3], [x1-x2_4, y1-y2_4]])
    MM = cv2.getPerspectiveTransform(pts1,pts2)
    imgout_p = cv2.warpPerspective(img,MM,(x1,y1),2,0,1)
    imgout = imgout_p[min(y2_1,y2_2):y1 -min(y2_3,y2_4),min(x2_1,x2_3):x1-min(x2_2,x2_4)]
    imgout = cv2.resize(imgout,(y1,x1),interpolation=cv2.INTER_CUBIC)
    return imgout

def pasteWord_easy(word,name,fontdir,BGR):
    '''输入一个文字，输出一张包含该文字的图片'''
    pygame.init()
    font = pygame.font.Font(fontdir, 32)#24
    text = word#.decode('utf-8')
    rtext = font.render(text, True, (0, 0, 0),(255, 255, 255))
    img = pygame.surfarray.pixels2d(rtext)
    img = 255 - np.transpose(img)
    img = cv2.resize(img,(250,250),interpolation=cv2.INTER_CUBIC)
    image = np.zeros((250,250,3))
    image[:,:,0] = cv2.add(img,BGR[0])
    image[:,:,1] = cv2.add(img,BGR[1])
    image[:,:,2] = cv2.add(img,BGR[2])

    imgout = perspective_trans(image)
    return imgout
    # imgName = "./creatdata/" + name+ ".png"
    # cv2.imencode('.png',imgout)[1].tofile(imgName)

def pasteWord_easy(word,name,fontdir):
    '''输入一个文字，输出一张包含该文字的图片'''
    pygame.init()
    font = pygame.font.Font(fontdir, 32)#24
    text = word#.decode('utf-8')
    rtext = font.render(text, True, (0, 0, 0),(255, 255, 255))
    img = pygame.surfarray.pixels2d(rtext)
    img = 255 - np.transpose(img)
    img = cv2.resize(img,(250,250),interpolation=cv2.INTER_CUBIC)
    imgout = perspective_trans(img)
    return imgout

def getRandomBGimg(bgImgpath,shape):
    for root,dirs,files in os.walk(bgImgpath):
        index = np.random.randint(0,len(files))
        bgimgname = os.path.join(root,files[index])
        img = cv2.imdecode(np.fromfile(bgimgname,dtype=np.uint8),-1)
        # if img.ndim==3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgbg = cv2.adaptiveThreshold(img,255,ADAPTIVE_THRESH_GAUSSIAN_C,31,5)
        # imgout =
        rowsta,colsta = shape[0:2]
        pic_row,pic_col = img.shape[0:2]
        # if rowsta>pic_row or colsta> pic_col:
        #     return None
        row_start = np.random.randint(0,pic_row - rowsta)
        col_start = np.random.randint(0,pic_col - colsta)
        bgcut = img[row_start:row_start+rowsta,col_start:col_start+colsta]
        return bgcut

def getChineseWordsPicture():
    line, fonts = readChineseGB1_and_fonts("num.txt", "./fonts/fonts.txt")
    # print(line[2108])
    for word in line:
        if word =="\n":
            continue
        print(word)
        for num in range(1000):
            font = random_fonts(fonts)
            print(word)
            name = str(word)+str(num)
            pasteWord_easy(word,name, font,[45,53,72])

def generateBGword(labelFile,fontsFile,bgpath,outpath,Num):
    line, fonts = readChineseGB1_and_fonts(labelFile, fontsFile)
    for word in line:
        if word =="\n":
            continue
        print(word)
        ImgOutPath = os.path.join(outpath,word)
        make_dir(ImgOutPath)
        for num in range(Num):
            font = random_fonts(fonts)
            # print(word)
            name = str(word)+str(num)
            img = pasteWord_easy(word,name, font)
            img= cv2.resize(img,(40,40),interpolation=cv2.INTER_CUBIC)
            ErodeFlag = np.random.randint(0,11)
            if ErodeFlag%5==0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                img = cv2.erode(img, kernel)
            rows,cols = img.shape[0:2]
            bgimg =  getRandomBGimg(bgpath,[rows,cols])
            fgcut = img
            bgcut_g = cv2.cvtColor(bgimg, cv2.COLOR_BGR2GRAY)
            th = (0.8-0.2)*np.random.random_sample()+0.2
            normal_clone = th*bgcut_g+(1.0-th)*fgcut
            imgName = ImgOutPath+"/" + name+ ".png"
            cv2.imencode('.png',normal_clone)[1].tofile(imgName)



if __name__=="__main__":
    labelFile = "num.txt"
    fontsFile = "./fonts/fonts.txt"
    bgpath = "./bgImg/"
    outpath = "./chinese3500/"
    Num = 200
    generateBGword(labelFile,fontsFile,bgpath,outpath,Num)
