//
//  main.cpp
//  opencv_FLANNfeatureMatching --- the feature matching with FLANN
//
//  Created by ZHHJemotion on 2016/11/22.
//  Copyright © 2016年 Lukas_Zhang. All rights reserved.
//

#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <string>

#define PATH string("/Users/zhangxingjian/Desktop/Programming/C++/OpenCV/opencv_FLANNfeatureMatching/opencv_FLANNfeatureMatching/opencv_FLANNfeatureMatching/")

using namespace cv;
using namespace std;

//全局函数声明
static void showHelpText(); //显示帮助文字

// ------------------------- main() function -------------------------------
int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    
    showHelpText();
    
    //载入源图像
    Mat srcImage1 = imread(PATH+string("1.jpg"),1);
    Mat srcImage2 = imread(PATH+string("2.jpg"),1);
    //异常处理
    if (!srcImage1.data || !srcImage2.data)
    {
        printf("Error in reading image!!!\n");
        return false;
    }
    
    //用 SURF 算子检测特征关键点
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    std::vector<KeyPoint> keyPoint1, keyPoint2;
    
    //调用 detect 函数检测 SURF 特征关键点，保存在 vector 容器内
    detector.detect(srcImage1, keyPoint1);
    detector.detect(srcImage2, keyPoint2);
    
    //计算描述符
    SurfDescriptorExtractor extractor;
    Mat descriptor1, descriptor2;
    extractor.compute(srcImage1, keyPoint1, descriptor1);
    extractor.compute(srcImage2, keyPoint2, descriptor2);
    
    //用 FLANN 进行匹配
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    
    double maxDist = 0;
    double minDist = 100;
    
    //快速计算特征关键点之间的最大和最小距离
    for (int i = 0; i < descriptor1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < minDist)
        {
            minDist = dist;
        }
        if (dist > maxDist)
        {
            maxDist = dist;
        }
    }
    
    printf("The Max Dist: %f \n", maxDist);
    printf("The Min Dist: %f \n", minDist);
    
    //只绘制“good”匹配
    std::vector<DMatch> goodMatches;
    
    for (int i = 0; i < descriptor1.rows; i++)
    {
        if (matches[i].distance <= max(2*minDist, 0.02)) // why 0.02 ?????? 这里选择good的匹配
        {
            goodMatches.push_back(matches[i]);
        }
    }
    
    Mat matchesImage;
    drawMatches(srcImage1, keyPoint1, srcImage2, keyPoint2, goodMatches, matchesImage);
    
    imshow("Good Matches", matchesImage);
    
    for (int i = 0; i < (int)goodMatches.size(); i++)
    {
        printf("The Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d \n", i, goodMatches[i].queryIdx, goodMatches[i].trainIdx);
    }
    
    imwrite(PATH+string("Good_Matches.jpg"), matchesImage);
    
    waitKey(0);
    
    return 0;
}

// --------------------- showHelpText() function ------------------------
static void showHelpText()
{
    //输出一些帮助信息
    printf("\n\n\n\t欢迎来到【SURF特征描述--FLANN】\n\n");
    std::cout<<"\t当前使用的OpenCV版本为 OpenCV"<<CV_VERSION;
    printf("\n\n\t\t\t\t\t\t\t\t by ZHHJemotion\n\n\n");
}

