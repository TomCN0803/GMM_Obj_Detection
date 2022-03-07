#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

// 高斯单模型中的参数
struct GaussianDistribution {
    float w;      // 权重
    float u;      // 期望
    float sigma;  // 方差
};

// 每个像素的GMM模型
struct PixelGMM {
    int currentGMMs;            // 当前像素点已经有的GMM数量
    GaussianDistribution *gd;   // 所含有的高斯模型数组
};

const int GAUSSIAN_MODULE_NUMS = 5; // 模型最大数量
const float ALPHA = 0.005;          // 学习率
const float SIGMA = 20;             // 标准差初始值
const float WEIGHT = 0.05;          // 高斯模型权重初始值
const float T = 0.7;                // 有效高斯分布阈值
const float LAMBDA = 2.5;
const int HISTORY = 120;             // 训练的帧数

class MOG {
private:
    int history;    // 需要训练的帧数
    int trained;    // 已经训练的帧数
    bool modelInit;     // 模型是否已经初始化
    PixelGMM *pixelGMM;      // 每个像素的GMM模型

public:
    void apply(Mat &image, Mat &foregroundMask);   // 主要算法函数

    void setHistory(int history);  // 设置训练帧数

    MOG();  // 构造函数
};

MOG::MOG() {
    history = 0;
    trained = 0;
    modelInit = false;
    pixelGMM = nullptr;
}

void MOG::apply(Mat &image, Mat &foregroundMask) {
    int rows = image.rows;
    int cols = image.cols;
    this->trained++;

    // 模型初始化，初始化各个像素的高斯分布
    if (!(this->modelInit)) {
        // 为像素点高斯模型分配空间
        this->pixelGMM = new PixelGMM[rows * cols * sizeof(PixelGMM)];
        for (auto i = 0; i < rows * cols; i++) {
            this->pixelGMM[i].currentGMMs = 0;
            this->pixelGMM[i].gd = new GaussianDistribution[GAUSSIAN_MODULE_NUMS * sizeof(GaussianDistribution)];
        }

        // 初始化高斯分布的参数
        for (auto i = 0; i < rows; i++) {
            for (auto j = 0; j < cols; j++) {
                for (auto n = 0; n < GAUSSIAN_MODULE_NUMS; n++) {
                    this->pixelGMM[i * cols + j].gd[n].w = 0;
                    this->pixelGMM[i * cols + j].gd[n].u = 0;
                    this->pixelGMM[i * cols + j].gd[n].sigma = SIGMA;
                }
            }
        }

        this->modelInit = true;
    }

    if (this->trained > this->history) {
        foregroundMask.create(rows, cols, CV_8UC1);
        foregroundMask.setTo(Scalar(0));
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int matchedIndex = -1;      // 最终匹配的高斯模型索引
            unsigned char pixel = image.at<unsigned char>(i, j);
            int pixelPos = i * cols + j;    // 当前像素点的坐标
            GaussianDistribution *pixelModel = pixelGMM[pixelPos].gd;   // 当前像素点的高斯多模型

            for (int m = 0; m < pixelGMM[pixelPos].currentGMMs; ++m) {
                if (fabs(pixel - pixelModel[m].u) < LAMBDA * pixelModel[m].sigma) {
                    // 更新权重
                    if (pixelModel[m].w > 1)
                        pixelModel[m].w = 1;
                    else
                        pixelModel[m].w = (1 - ALPHA) * pixelModel[m].w + ALPHA;

                    // 更新期望
                    pixelModel[m].u = (1 - ALPHA) * pixelModel[m].u + ALPHA * pixel;

                    // 更新标准差
                    pixelModel[m].sigma = sqrt((1 - ALPHA) * pixelModel[m].sigma * pixelModel[m].sigma +
                                               ALPHA * (pixel - pixelModel[m].u) * (pixel - pixelModel[m].u));

                    // 根据sort_key进行排序
                    int n;
                    for (n = m - 1; n >= 0; n--) {
                        if (pixelModel[n].w / pixelModel[n].sigma < pixelModel[n + 1].w / pixelModel[n + 1].sigma) {
                            float temp;
                            temp = pixelModel[n].sigma;
                            pixelModel[n].sigma = pixelModel[n + 1].sigma;
                            pixelModel[n + 1].sigma = temp;
                            temp = pixelModel[n].u;
                            pixelModel[n].u = pixelModel[n + 1].u;
                            pixelModel[n + 1].u = temp;
                            temp = pixelModel[n].w;
                            pixelModel[n].w = pixelModel[n + 1].w;
                            pixelModel[n + 1].w = temp;
                        } else
                            break;
                    }
                    matchedIndex = n + 1;
                    break;
                } else
                    pixelModel[m].w *= 1 - ALPHA;
            }

            // 增加新的模型
            if (matchedIndex == -1) {
                if (pixelGMM[pixelPos].currentGMMs == GAUSSIAN_MODULE_NUMS) {
                    pixelModel[pixelGMM[pixelPos].currentGMMs - 1].sigma = SIGMA;
                    pixelModel[pixelGMM[pixelPos].currentGMMs - 1].u = pixel;
                    pixelModel[pixelGMM[pixelPos].currentGMMs - 1].w = WEIGHT;
                } else {
                    pixelModel[pixelGMM[pixelPos].currentGMMs].sigma = SIGMA;
                    pixelModel[pixelGMM[pixelPos].currentGMMs].u = pixel;
                    pixelModel[pixelGMM[pixelPos].currentGMMs].w = WEIGHT;
                    pixelGMM[pixelPos].currentGMMs++;
                }
            }

            if (trained > history) {
                // 高斯分布权值归一化
                float weightDen = 0;
                for (int n = 0; n < pixelGMM[pixelPos].currentGMMs; n++) {
                    weightDen += pixelModel[n].w;
                }

                float weightSum = 0;
                int foreground = -1;    // 属于前景模型的高斯索引
                for (int n = 0; n < pixelGMM[pixelPos].currentGMMs; n++) {
                    weightSum += pixelModel[n].w / weightDen;
                    if (weightSum > T) {
                        foreground = n + 1;
                        break;
                    }
                }

                // 属于前景点的判断条件
                if ((matchedIndex >= foreground) || matchedIndex == -1) {
                    foregroundMask.at<unsigned char>(i, j) = 255;
                }
            }
        }
    }
}

void MOG::setHistory(int history) {
    this->history = history;
}


int main() {
    Mat frame, frameGrayScale, foregroundMask;
    MOG MOGSubtractor;
    VideoCapture cap("/Users/tomcn0803/CLionProjects/ObjectDetection/Test01.avi");
//    VideoCapture cap("/Users/tomcn0803/CLionProjects/ObjectDetection/Walk1.mpg");
//    VideoCapture cap("/Users/tomcn0803/CLionProjects/ObjectDetection/Browse3.mpg");
//    VideoCapture cap("/Users/tomcn0803/CLionProjects/ObjectDetection/Meet_WalkTogether1.mpg");
    if (!cap.isOpened()) {
        cout << "无法打开文件！" << endl;
        return -1;
    }

    MOGSubtractor.setHistory(HISTORY);
    namedWindow("Target Video");
    while (true) {
        if (!cap.read(frame)) break;

        cvtColor(frame, frameGrayScale, COLOR_BGR2GRAY);
        GaussianBlur(frameGrayScale, frameGrayScale, Size(5, 5), 0, 0);

        MOGSubtractor.apply(frameGrayScale, foregroundMask);
        if (!foregroundMask.empty())
            imshow("fgmask", foregroundMask);
        else
            continue;

        // 轮廓检测
        vector<vector<Point>>(contours);
        Scalar color = Scalar(0, 255, 0);
        findContours(foregroundMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<Rect> boundRect(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            if (arcLength(contours[i], true) >= 100) {
                boundRect[i] = boundingRect(Mat(contours[i]));
                rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2);
            }
        }

        imshow("result", frame);    // 展示最终结果

        if (waitKey(100) == 27)
            break;
    }

    return 0;
}
