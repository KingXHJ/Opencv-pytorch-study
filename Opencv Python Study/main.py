# import cv2 as cv
# # print opencv version
# print(cv.__version__)
#
# # read picture
# img1 = cv.imread('test.jpg', cv.IMREAD_COLOR)
# img0 = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)
# imgneg1 = cv.imread('test.jpg', cv.IMREAD_UNCHANGED)
#
# # # show pictures
# # cv.imshow('RGB',img1)
# # cv.imshow('gray',img0)
# # cv.imshow('alpha',imgneg1)
#
# cv.namedWindow('RGB', cv.WINDOW_AUTOSIZE) # 全屏窗口后，会有灰色的边框
# # cv.namedWindow('RGB',cv.WINDOW_NORMAL) # 全屏窗口后，图像会自动跟着拉伸
#
# cv.imshow('RGB', img1)
# # Indefinitely wait for a press of ant key
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # save picture
# cv.imwrite('test.png', img1)
#
#
#
#


# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.png', 1)
# cv.namedWindow('image_test',cv.WINDOW_NORMAL)
# cv.imshow('image_test', img)
# key = cv.waitKey(0) & 0XFF
# # 查阅资料我才知道，原来系统中按键对应的ASCII码值并不一定仅仅只有8位，同一按键对应的ASCII并不一定相同（但是后8位一定相同）
# # 为什么会有这个差别？是系统为了区别不同情况下的同一按键。
# # 比如说“q”这个按键
# # 当小键盘数字键“NumLock”激活时，“q”对应的ASCII值为100000000000001100011 。
# # 而其他情况下，对应的ASCII值为01100011。
# # 相信你也注意到了，它们的后8位相同，其他按键也是如此。
# # 为了避免这种情况，引用&0xff，正是为了只取按键对应的ASCII值后8位来排除不同按键的干扰进行判断按键是什么。
# if key == 23:
#     cv.destroyAllWindows()
# elif key == ord('s'):
#     cv.imwrite('test1.png', img)
#     cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0) # 换成本地视频的路径
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     ret,frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = np.zeros((512, 512, 3), np.uint8)
# cv.line(img, (0, 0), (511, 511), (255, 0, 0), 1, cv.LINE_AA, 2)
#
# cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
#
# cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
#
# cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
#
# pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv.polylines(img, [pts], True, (0, 255, 255))
#
# font = cv.FONT_HERSHEY_COMPLEX
# cv.putText(img, 'KingXHJ', (10, 400), font, 2, (255, 255, 255), 2, cv.LINE_AA)
#
# cv.imshow('test', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 列出所有可用的可用事件
# import cv2 as cv
# events = [i for i in dir(cv) if 'EVENT' in i]
# print(events)

# # mouse operation
# import numpy as np
# import cv2 as cv
#
# # 创建鼠标回调函数具有特定的格式，该格式在所有地方都相同
# # callback function
# def draw_circle(event, x, y, flags, param):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img, (x, y), 100, (255, 0, 0), -1)
#
#
# img = np.zeros((512, 512, 3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image', draw_circle)
# while (1):
#     cv.imshow('image', img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
# cv.destroyAllWindows()

# mouse operation advance
# import numpy as np
# import cv2 as cv
#
# drawing = False # 如果按下鼠标，则为真
# mode = True # 如果为真，绘制矩形。按 m 键可以切换到曲线
# ix, iy = -1, -1
#
#
# def draw_circle(event, x, y, flags, param):
#     global ix, iy, drawing, mode
#     if event == cv.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#     elif event == cv.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv.circle(img,(x,y),5,(0,0,255),-1)
#     elif event == cv.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv.circle(img,(x,y),5,(0,0,255),-1)
#
# # 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#      cv.imshow('image',img)
#      if cv.waitKey(20) & 0xFF == 27:
#         break
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
#
# def nothing(x):
#     pass
#
#
# img = np.zeros((300, 512, 3), np.uint8)
# cv.namedWindow('image')
#
# cv.createTrackbar('R', 'image', 0, 255, nothing)
# cv.createTrackbar('G', 'image', 0, 255, nothing)
# cv.createTrackbar('B', 'image', 0, 255, nothing)
#
# switch = '0 : OFF \n 1 : ON'
# cv.createTrackbar(switch, 'image', 0, 1, nothing)
# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k ==27:
#         break
#     r=cv.getTrackbarPos('R','image')
#     g = cv.getTrackbarPos('G', 'image')
#     b = cv.getTrackbarPos('B', 'image')
#     s = cv.getTrackbarPos(switch, 'image')
#     if s == 0:
#         img[:]=0
#     else:
#         img[:]=[b,g,r]
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
#
# px = img[100, 100]
# print(px)
#
# # blue = img[100, 100, 0]
# # blue = img[100, 100, 1]
# blue = img[100, 100, 2]
# print(blue)
#
# img[100, 100] = [255, 255, 255]
# print(img[100, 100])
#
# img.item(10, 10, 2)
# print(img[10, 10, 2])
#
# img.itemset((10, 10, 2), 100)
# print(img[10, 10, 2])
#
# print(img.shape)
# print(img.size)
# print(img.dtype)
#
# something = img[200:400, 200:400]
# img[500:700, 1000:1200] = something
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# b, g, r = cv.split(img)
# cv.imshow('b', b)
# cv.imshow('g', g)
# cv.imshow('r', r)
# cv.waitKey(0)
# cv.destroyAllWindows()
# img = cv.merge((b, g, r))
#
# b = img[:, :, 0] = 0

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# BLUE = [255, 0, 0]
# img1 = cv.imread('test.png')
# replicate = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REPLICATE)
# reflect = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REFLECT)
# reflect101 = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
# wrap = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_WRAP)
# constant = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)
# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
# plt.show()

# import numpy as np
# import cv2 as cv
#
# x = np.uint8([250])
# y = np.uint8([10])
# print(cv.add(x, y))  # OpenCV加法是饱和运算
# print(x + y)  # Numpy加法是模运算
#
# img1 = cv.imread('test.jpg')
# print(img1.shape)
# img2 = cv.imread('test1.jpg')
# dst = cv.addWeighted(img1, 0.2, img2, 0.8, 0)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
#
# e1 = cv.getTickCount()
#
# img1 = cv.imread('test.jpg')
# img2 = cv.imread('test2.jpg')
# rows, cols, channels = img2.shape
# roi = img1[0:rows, 0:cols]
# img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)  # 彩色的地方是非0
# mask_inv = cv.bitwise_not(mask)  # 彩色的地方是0
# img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv.bitwise_and(img2, img2, mask=mask)
# dst = cv.add(img1_bg, img2_fg)
# img1[0:rows, 0:cols] = dst
# cv.imshow('res', img1)
#
# e2 = cv.getTickCount()
#
# t = (e2 - e1) / cv.getTickFrequency()
# print(t)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(flags)

# import cv2 as cv
# import numpy as np
#
# cap = cv.VideoCapture(0)
# while (1):
#     _, frame = cap.read()
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#
#     mask = cv.inRange(hsv, lower_blue, upper_blue)
#
#     res = cv.bitwise_and(frame,frame,mask=mask)
#
#     cv.imshow('frame',frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()
import cv2
import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# height, width = img.shape[:2]
# res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
# cv.imshow('source', img)
# cv.imshow('scale', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# rows, cols = img.shape
# print(img.shape)
#
# M1 = np.float32([[1, 0, 100], [0, 1, 50]])
# dst1 = cv.warpAffine(img, M1, (cols, rows))
# # cols-1 和 rows-1 是坐标限制
# M2 = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
# dst2 = cv.warpAffine(img, M2, (cols, rows))
#
# img3 = cv.imread('test.jpg')
# rows, cols, channels = img3.shape
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M3 = cv.getAffineTransform(pts1, pts2)
# dst3 = cv.warpAffine(img3, M3, (cols, rows))
# cv.imshow('img1', dst1)
# cv.imshow('img2', dst2)
# cv.imshow('img3', img3)
# cv.imshow('dst3', dst3)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.png')
# rows, cols, ch = img.shape
# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
# M = cv.getPerspectiveTransform(pts1, pts2)
# dst = cv.warpPerspective(img, M, (300, 300))
# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv.imread('test.jpg',0)
# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks()
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# blur = cv.GaussianBlur(img, (5, 5), 0)
# ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv2.THRESH_OTSU)
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
#           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
#           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
# for i in range(3):
#     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
#     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
#     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
#     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
# plt.show()
#
# # Otsu
# img = cv.imread('test.jpg', 0)
# blur = cv.GaussianBlur(img, (5, 5), 0)
# hist = cv.calcHist([blur], [0], None, [256], [0, 256])
# hist_norm = hist.ravel() / hist.max()
# Q = hist_norm.cumsum()
# bins = np.arange(256)
# fn_min = np.inf
# thresh = -1
# for i in range(1, 256):
#     p1, p2 = np.hsplit(hist_norm, [i])  # 概率
#     q1, q2 = Q[i], Q[255] - Q[i]  # 对类求和
#     b1, b2 = np.hsplit(bins, [i])  # 权重
#     # 寻找均值和方差
#     m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
#     v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
#     # 计算最小化函数
#     fn = v1 * q1 + v2 * q2
#     if fn < fn_min:
#         fn_min = fn
#     thresh = i
# # 使用OpenCV函数找到otsu的阈值
# ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# print("{} {}".format(thresh, ret))

# import numpy as np
# import  cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('test.jpg')
# kernel = np.ones((5,5),np.float32)/25
# dst = cv.filter2D(img,-1,kernel)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# # blur = cv.blur(img, (5, 5))
# # blur = cv.GaussianBlur(img, (5, 5), 0)
# # blur = cv.medianBlur(img, 5) # 消除椒盐噪声
# blur = cv.bilateralFilter(img, 9, 75, 75)
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

# import cv2 as cv
# import numpy as np
#
# img = cv.imread('test.jpg', 0)
# kernel = np.ones((5, 5), np.uint8)
# erosion = cv.erode(img, kernel, iterations=1)  # 侵蚀白色
# dilation = cv.dilate(img, kernel, iterations=1)  # 膨胀黑色
# opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # 开运算，先侵蚀再膨胀，用于去白色噪
# closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)  # 闭运算，先膨胀再侵蚀，用于去黑色噪声
# gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)  # 膨胀图与腐蚀图之差
# tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)  # 原始图像与开运算之差
# blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)  # 原图像与闭运算之差
# cv.imshow('img', img)
# cv.imshow('erosion', erosion)
# cv.imshow('dilation', dilation)
# cv.imshow('opening', opening)
# cv.imshow('closing', closing)
# cv.imshow('gradient', gradient)
# cv.imshow('tophat', tophat)
# cv.imshow('blackhat', blackhat)
#
# print(cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
# print(cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
# print(cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# laplacian = cv.Laplacian(img, cv.CV_64F)
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# # 在我们的最后一个示例中，输出数据类型为 cv.CV_8U 或 np.uint8 。但这有一个小问题。黑色到
# # 白色的过渡被视为正斜率（具有正值），而白色到黑色的过渡被视为负斜率（具有负值）。因
# # 此，当您将数据转换为np.uint8时，所有负斜率均设为零。简而言之，您会错过这一边缘信息。
# # 如果要检测两个边缘，更好的选择是将输出数据类型保留为更高的形式，例如 cv.CV_16S ，
# # cv.CV_64F 等，取其绝对值，然后转换回 cv.CV_8U 。 下面的代码演示了用于水平Sobel滤波器和
# # 结果差异的此过程。
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.png', 0)
# # Output dtype = cv.CV_8U
# sobelx8u = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
# # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
# sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('test.jpg',0)
# edges = cv.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# import cv2 as cv
# import numpy as np
# # 先降低分辨率，再恢复分辨率，一定会丢失信息
# img = cv.imread('test.jpg')
# lower_reso = cv.pyrDown(img)
# higher_reso = cv.pyrUp(img)
# cv.imshow('source', img)
# cv.imshow('pyrDown', lower_reso)
# cv.imshow('pyrUp', higher_reso)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np, sys
#
# A = cv.imread('test.jpg')
# B = cv.imread('test1.jpg')
# # 生成A的高斯金字塔
# G = A.copy()
# gpA = [G]
# for i in range(6):
#     G = cv.pyrDown(G)
#     gpA.append(G)
# # 生成B的高斯金字塔
# G = B.copy()
# gpB = [G]
# for i in range(6):
#     G = cv.pyrDown(G)
#     gpB.append(G)
# # 生成A的拉普拉斯金字塔
# lpA = [gpA[5]]
# for i in range(5, 0, -1):
#     GE = cv.pyrUp(gpA[i])
#     L = cv.subtract(gpA[i - 1], GE)
#     lpA.append(L)
# # 生成B的拉普拉斯金字塔
# lpB = [gpB[5]]
# for i in range(5, 0, -1):
#     GE = cv.pyrUp(gpB[i])
#     L = cv.subtract(gpB[i - 1], GE)
#     lpB.append(L)
# # 现在在每个级别中添加左右两半图像
# LS = []
# for la, lb in zip(lpA, lpB):
#     rows, cols, dpt = la.shape
#     ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
#     LS.append(ls)
# # 现在重建
# ls_ = LS[0]
# for i in range(1, 6):
#     ls_ = cv.pyrUp(ls_)
#     ls_ = cv.add(ls_, LS[i])
# # 图像与直接连接的每一半
# real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))
# cv.imshow('Pyramid_blending2.jpg', ls_)
# cv.imshow('Direct_blending.jpg', real)

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imggray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0,255,0), 3)
# # cv.drawContours(img, contours, 3, (0,255,0), 3)
# # cnt = contours[4]
# # cv.drawContours(img, [cnt], 0, (0,255,0), 3)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# ret, thresh = cv.threshold(img, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv.moments(cnt)
# cx = int(M['m10'] / M['m00'])
# cy = int(M['m01'] / M['m00'])
# area = cv.contourArea(cnt)
# perimeter = cv.arcLength(cnt, True)
#
# epsilon = perimeter * 0.1
# approx = cv.approxPolyDP(cnt, epsilon, True)
# img1 = cv.imread('test.jpg', 1)
# dst = cv.drawContours(img1, approx, -1, (0, 255, 0), 3)
#
# hull = cv.convexHull(cnt)
# dst2 = cv.drawContours(img1, hull, -1, (255, 0, 0), 3)
#
# k = cv.isContourConvex(cnt)
#
# x, y, w, h = cv.boundingRect(cnt)
# bound_rec = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# rect = cv.minAreaRect(cnt)
# box = cv.boxPoints(rect)
# box = np.int0(box)
# dst3 = cv.drawContours(img, [box], 0, (0, 0, 255), 2)
#
# (x, y), radius = cv.minEnclosingCircle(cnt)
# center = (int(x), int(y))
# radius = int(radius)
# dst4 = cv.circle(img, center, radius, (0, 255, 0), 2)
#
# ellipse = cv.fitEllipse(cnt)
# dst5 = cv.ellipse(img, ellipse, (0, 255, 0), 2)
#
# rows, cols = img.shape[:2]
# [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
# lefty = int((-x * vy / vx) + y)
# righty = int(((cols - x) * vy / vx) + y)
# dst6 = cv.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
#
# x, y, w, h = cv.boundingRect(cnt)
# aspect_ratio = float(w) / h
#
# area = cv.contourArea(cnt)
# x, y, w, h = cv.boundingRect(cnt)
# rect_area = w * h
# extent = float(area) / rect_area
#
# area = cv.contourArea(cnt)
# hull = cv.convexHull(cnt)
# hull_area = cv.contourArea(hull)
# solidity = float(area) / hull_area
#
# area = cv.contourArea(cnt)
# equi_diameter = np.sqrt(4 * area / np.pi)
#
# (x, y), (MA, ma), angle = cv.fitEllipse(cnt)
#
# mask = np.zeros(img.shape, np.uint8)
# cv.drawContours(mask, [cnt], 0, 255, -1)
# pixelpoints = np.transpose(np.nonzero(mask))
#
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img, mask=mask)
#
# mean_val = cv.mean(img, mask=mask)
#
# leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
# rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
# topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
# bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
#
# hull = cv.convexHull(cnt, returnPoints=False)
# defects = cv.convexityDefects(cnt, hull)
#
# print('M:', M)
# print('cx:', cx)
# print('cy:', cy)
# print('area:', area)
# print('perimeter:', perimeter)
# print('k:', k)
# print('aspect_ratio', aspect_ratio)
# print('extent', extent)
# print('solidity', solidity)
# print('equi_diameter', equi_diameter)
# print('pixelpoints', pixelpoints)
#
# cv.imshow('approx', dst)
# cv.imshow('hull', dst2)
# cv.imshow('bound_rec', bound_rec)
# cv.imshow('box', dst3)
# cv.imshow('circle', dst4)
# cv.imshow('ellipse', dst5)
# cv.imshow('line', dst6)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
#
# img = cv.imread('test.jpg')
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(img_gray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, 2, 1)
# cnt = contours[0]
# hull = cv.convexHull(cnt, returnPoints=False)
# defects = cv.convexityDefects(cnt, hull)
# for i in range(defects.shape[0]):
#     s, e, f, d = defects[i, 0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])
#     cv.line(img, start, end, [0, 255, 0], 2)
#     cv.circle(img, far, 5, [0, 0, 255], -1)
#
# dist = cv.pointPolygonTest(cnt, (50, 50), True)
#
# print('dist', dist)
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
#
# img1 = cv.imread('test.jpg', 0)
# img2 = cv.imread('test1.jpg', 0)
# ret1, thresh1 = cv.threshold(img1, 127, 255, 0)
# ret2, thresh2 = cv.threshold(img2, 127, 255, 0)
# contours1, hierarchy1 = cv.findContours(thresh1, 2, 1)
# cnt1 = contours1[0]
# # contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_LIST, 1)
# # contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_EXTERNAL, 1)
# # contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_CCOMP, 1)
# contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_TREE, 1)
# cnt2 = contours2[0]
# ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
# print(ret)  # 结果越低，匹配越好
# print(hierarchy2)

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# # img = cv.imread('test.jpg', 0)
# # hist = cv.calcHist([img], [0], None, [256], [0, 256])
# # hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# # hist = np.bincount(img.ravel(), minlength=256)
#
# # img.ravel()–把多维数组转化成一维数组，指向原数组的地址
# # img。flatten()-和ravel的作用几乎相同，但是是指向原数组拷贝的地址
# # plt.hist(img.ravel(),256,[0,256]); plt.show()
#
# # img = cv.imread('test.jpg')
# # color = ('b', 'g', 'r')
# # for i, col in enumerate(color):
# #     histr = cv.calcHist([img], [i], None, [256], [0, 256])
# #     plt.plot(histr,color = col)
# #     plt.xlim([0,256])
# # plt.show()
#
# img = cv.imread('test.jpg', 0)
# mask = np.zeros(img.shape[:2], np.uint8)
# # print(img.shape)
# mask[100:300, 100:400] = 255
# masked_img = cv.bitwise_and(img, img, mask=mask)
# hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
# hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()

# # 一幅好的图像会有来自图像所有区域的像素。因此，您需要将这个直方图拉伸到两端
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# # CDF 累积分布函数(cumulative distribution function)
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
#
# cdf1_m = np.ma.masked_equal(cdf, 0)
# cdf1_m = (cdf1_m - cdf1_m.min()) * 255 / (cdf1_m.max() - cdf1_m.min())
# cdf1 = np.ma.filled(cdf1_m, 0).astype('uint8')
# img2 = cdf1[img]
# hist1, bins1 = np.histogram(img2.flatten(), 256, [0, 256])
# cdf2 = hist1.cumsum()
# cdf2_normalized = cdf2 * float(hist1.max()) / cdf2.max()
#
# plt.subplot(211)
# plt.plot(cdf_normalized, color='b')
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
#
# plt.subplot(212)
# plt.plot(cdf1, color='b')
# plt.hist(img2.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
# plt.show()

# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# equ = cv.equalizeHist(img)
# res = np.hstack((img, equ))
# cv.imshow('img', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 为了解决上述2个问题, 就有2方面的解决方法: 一是解决全局性问题, 二是解决背景噪声增强问题.
# #
# # 针对全局性问题: 有人提出了对图像分块的方法, 每块区域单独进行直方图均衡, 这样就可以利用局部信息来增强图像, 这样就可以解决全局性问题;
# # 针对背景噪声增强问题: 主要背景增强太过了, 因而有人提出了对对比度进行限制的方法, 这样就可以解决背景噪声增强问题;
# # 将上述二者相结合就是 CLAHE 方法, 其全称为: Contrast Limited Adaptive Histogram Equalization.
# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cll = clahe.apply(img)
# cv.imshow('img', img)
# cv.imshow('cll', cll)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# # hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]]
# plt.imshow(hist, interpolation='nearest')
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# roi = cv.imread('test.jpg')
# hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
#
# target = cv.imread('test1.jpg')
# hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
#
# M = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# I = cv.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
#
# R = M / I
# h, s, v = cv.split(hsvt)
# B = R[h.ravel(), s.ravel()]
# B = np.minimum(B, 1)
# B = B.reshape(hsvt.shape[:2])
#
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# cv.filter2D(B, -1, disc, B)
# B = np.uint8(B)
# cv.normalize(B, B, 0, 255, cv.NORM_MINMAX)
#
# ret,thresh = cv.threshold(B,50,255,0)

# import numpy as np
# import cv2 as cv
#
# roi = cv.imread('test.jpg')
# roi = cv.pyrDown(roi)
# hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# target = cv.imread('test1.jpg')
# target = cv.pyrDown(target)
# hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
#
# roihist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
#
# cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
# dst = cv.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
#
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# cv.filter2D(dst, -1, disc, dst)
#
# ret, thresh = cv.threshold(dst, 50, 255, 0)
# thresh = cv.merge((thresh, thresh, thresh))
# res = cv.bitwise_and(target, thresh)
# res = np.vstack((target, thresh, res))
# cv.imshow('res', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
#
# rows, cols = img.shape
# crow, ccol = rows // 2, cols // 2
# fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.real(img_back)
#
# # plt.subplot(121), plt.imshow(img, cmap='gray')
# # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# # plt.show()
#
# plt.subplot(131), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(img_back, cmap='gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()


# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
#
# # plt.subplot(121), plt.imshow(img, cmap='gray')
# # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122), plt.imshow(magnitude_spectum, cmap='gray')
# # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# # plt.show()
#
# rows, cols = img.shape
# # / 在python3中是浮点数除法，// 是向下取整
# crow, ccol = rows // 2, cols // 2
#
# mask = np.zeros((rows, cols, 2), np.uint8)
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
#
# fshift = dft_shift * mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv.idft(f_ishift)
# img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_back, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# print("{} {}".format(rows, cols))
#
# nrows = cv.getOptimalDFTSize(rows)
# ncols = cv.getOptimalDFTSize(cols)
# print("{} {}".format(nrows, ncols))
#
# nimg = np.zeros((nrows, ncols))
# nimg[:rows, :cols] = img

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# mean_filter = np.ones((3, 3))
#
# x = cv.getGaussianKernel(5, 10)
# gaussian = x * x.T
#
# scharr = np.array([[-3, 0, 3],
#                    [-10, 0, 10],
#                    [-3, 0, 3]])
#
# sobel_x = np.array([[-1, 0, 1],
#                     [-2, 0, 2],
#                     [-1, 0, 1]])
#
# sobel_y = np.array([[-1, -2, -1],
#                     [0, 0, 0],
#                     [1, 2, 1]])
#
# laplacian = np.array([[0, 1, 0],
#                       [1, -4, 1],
#                       [0, 1, 0]])
#
# filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
# filter_name = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x']
# fft_filters = [np.fft.fft2(x) for x in filters]
# fft_shift = [np.fft.fftshift(y) for y in fft_filters]
# mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap='gray')
#     plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
# plt.show()


# # 方差匹配方法：匹配度越高，值越接近于0。
# #
# #                归一化方差匹配方法：完全匹配结果为0。
# #
# #                相关性匹配方法：完全匹配会得到很大值，不匹配会得到一个很小值或0。
# #
# #                归一化的互相关匹配方法：完全匹配会得到1， 完全不匹配会得到0。
# #
# #                相关系数匹配方法：完全匹配会得到一个很大值，完全不匹配会得到0，完全负相关会得到很大的负数。
# #
# #       （此处与书籍以及大部分分享的资料所认为不同，研究公式发现，只有归一化的相关系数才会有[-1,1]的值域）
# #
# #                归一化的相关系数匹配方法：完全匹配会得到1，完全负相关匹配会得到-1，完全不匹配会得到0。
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# img2 = img.copy()
# template = cv.imread('test1.jpg', 0)
# w, h = template.shape[::-1]
#
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
#            'cv.TM_SQDIFF_NORMED']
#
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth) # 执行字符串表达的内容
#
#     res = cv.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img, top_left, bottom_right, 255, 2)
#     plt.subplot(121), plt.imshow(res, cmap='gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122), plt.imshow(img, cmap='gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img_rgb = cv.imread('test.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('test1.jpg', 0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]): # 加星号把列表里的元素取出来，这样就不会是把整个列表看成一个元素了
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
# cv.imshow('res', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2 as cv
# import numpy as np
#
# img = cv.imread(cv.samples.findFile('test.png'))
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150, apertureSize=3)
# lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
# print(lines)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv.imshow('houghlines all', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# import cv2 as cv
# import numpy as np
# img = cv.imread(cv.samples.findFile('sudoku.png'))
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray,50,150,apertureSize = 3)
# lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
# for line in lines:
#  x1,y1,x2,y2 = line[0]
#  cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv.imwrite('houghlines probability',img)
# cv.waitKey(0)
# cv.destroyAllWindows()