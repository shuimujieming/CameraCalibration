#include <iostream>
#include "opencv2/opencv.hpp"
//相机标定程序声明
void CameraCal();

//主函数
int main() {
    CameraCal();
    return 0;
}
//相机标定
void CameraCal()
{
    std::string img_path = "../photo//*.jpg";//相机标定图片路径
    float image_zoom = 0.25f;//图片显示缩放比例
    int board_w = 12,board_h = 9;//每行角点数，每列角点数
    cv::Size board_sz = cv::Size(board_w, board_h);	//标定板大小
    int board_n = board_w * board_h;//角点总数

    std::vector<std::string> file_path;//图片文件路径保存的向量

    //读取文件夹内全部文件并输出为vector数据
    cv::glob(img_path, file_path, false);//是否递归查找（是否存在子文件夹需要查找）

    //输出显示文件路径
    for (int i = 0; i < file_path.size(); i++)
    {
        std::cout << "picture " << i + 1 << " path: " << file_path[i] << std::endl;
    }

    //图片数量保存
    int n_boards = file_path.size();//标定板图片数目

    //包含所有图片的坐标点
    //每一张图片的坐标点的集合
    std::vector<std::vector<cv::Point2f>> image_points;//图像坐标点
    std::vector<std::vector<cv::Point3f>> object_points;//物体坐标系（世界坐标系）

    //图片大小
    cv::Size image_size;

    //循环遍历图片角点
    for (int i = 0; i < n_boards; i++)
    {
        cv::Mat image; //本次图片
        //读取图片
        image = cv::imread(file_path[i]);

        if (image.empty())
        {
            std::cout<<"Input The Picture "<< file_path[i] <<" is Empty！"<<std::endl;
            continue;
        }

        //图片尺寸
        image_size = image.size();

        //输出显示图片的像素大小
        if(i == 0)
        {
            std::cout<<"picture size is "<<image_size<<std::endl;
        }

        //该图片中全部角点坐标
        std::vector<cv::Point2f> corners;

        //自带亚像素
        bool found = cv::findChessboardCornersSB(image, board_sz, corners);

        //未找到时输出错误信息则跳过后续操作
        if (!found)
        {
            std::cout << "Couldn't found correct corners!" << std::endl;
            continue;
        }

        //在原图上画出棋盘格角点
        cv::drawChessboardCorners(image, board_sz, corners, found);

        //上一次处理的时间戳
        //计算时间
        static double last_captured_timestamp = 0;
        //记录当前时间
        double timestamp = (double)clock() / CLOCKS_PER_SEC;
        //计算频率
        double frequence = timestamp - last_captured_timestamp;


        //找到全部角点
        if (found)
        {
            //将当前图片中的角点坐标保存
            image_points.push_back(corners);

            //保存世界坐标
            object_points.push_back(std::vector<cv::Point3f>());
            std::vector<cv::Point3f>& opts = object_points.back();
            opts.resize(board_n);
            for (int j = 0; j < board_n; j++)
            {
                //角点的真实世界坐标（x，y，0）
                opts[j] = cv::Point3f((float)(j % board_w ), (float)(j / board_w ), 0.f);
            }
            //输出当前图片角点查找时间
            //记录上次时间
            last_captured_timestamp = timestamp;
            std::cout << "Cost " << frequence << " s \t"<< "Collected our " << (int)image_points.size() << " of " << n_boards << " needed chessboard images\n";
            std::cout << file_path[i] << std::endl <<std::endl;
        }

        //缩放显示
        cv::Mat image_show;
        cv::resize(image,image_show,cv::Size(),image_zoom,image_zoom);
        cv::imshow("Calibration", image_show);

        if ((cv::waitKey(40) & 255) == 27)
        {
            return ;
        }
    }
    //for循环终止处
    //销毁标定板图片窗口
    cv::destroyWindow("Calibration");

    //输出正在标定
    std::cout<<"...Calibrating Camera Params..."<<std::endl;

    cv::Mat intrinsic_matrix; //相机内参
    cv::Mat distortion_coeffs;//畸变参数
    std::vector<cv::Mat> rvecs;//旋转向量
    std::vector<cv::Mat> tvecs;//平移向量

    //标定相机并记录误差值
    double err = cv::calibrateCamera(
            object_points,
            image_points,
            image_size,
            intrinsic_matrix,
            distortion_coeffs,
            rvecs, //rvecs
            tvecs, //tvecs
            cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

    double total_err = 0.0;//所有图像的平均误差的总和
    double single_err = 0.0;//每幅图像的平均误差
    for(int i=0;i<n_boards;i++)
    {
        std::vector<cv::Point2f> image_points2;//保存重新计算得到的投影点
        cv::projectPoints(object_points[i],rvecs[i],tvecs[i],intrinsic_matrix,distortion_coeffs,image_points2);
        //计算心得投影点和旧的投影点之间的误差
        std::vector<cv::Point2f> tempImagePoint = image_points[i];
        cv::Mat tempImagePointMat = cv::Mat(1,tempImagePoint.size(),CV_32FC2);
        cv::Mat image_points2Mat = cv::Mat(1,image_points2.size(),CV_32FC2);
        for(int j=0;j<tempImagePoint.size();j++)
        {
            image_points2Mat.at<cv::Vec2f>(0,j) = cv::Vec2f(image_points2[j].x,image_points2[j].y);
            tempImagePointMat.at<cv::Vec2f>(0,j) = cv::Vec2f(tempImagePoint[j].x,tempImagePoint[j].y);
        }
        single_err = norm(image_points2Mat,tempImagePointMat,cv::NORM_L2);
        total_err += single_err /= board_n;
        std::cout<<"The "<<i+1<<" error is "<<single_err<<" pixels "<<std::endl;
        //fout<<"第"<<i+1<<"幅图像的平均误差："<<single_err<<"像素"<<std::endl;
    }

    std::cout<<"total error "<<total_err<<std::endl;
    std::cout << " ***DONE!***\\nReprojection error is " << err<<std::endl;

    //输出相机参数矩阵以及畸变参数
    cv::FileStorage fs("camera_params.xml", cv::FileStorage::WRITE);


    fs  << "image_width" << image_size.width
        << "image_height"<< image_size.height
        << "camera_matrix" << intrinsic_matrix
        << "distortion_coefficients" << distortion_coeffs
        << "rotation_vectors" << rvecs
        << "translation_vectors" <<tvecs;

    fs.release();

    fs.open("camera_params.xml", cv::FileStorage::READ);

    std::cout << "\nimage width: " << (int)fs["image_width"];
    std::cout << "\nimage height: " << (int)fs["image_height"];

    //从文件中读取参数
    cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    //输出参数
    std::cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    std::cout << "\ndistortion coefficients:" << distortion_coeffs_loaded << std::endl;

    cv::Mat map1, map2;
    //https://blog.csdn.net/u013341645/article/details/78710740
    //计算无畸变和修正转换映射
    cv::initUndistortRectifyMap(
            intrinsic_matrix_loaded,
            distortion_coeffs_loaded,
            cv::Mat::eye(3,3,CV_32F),
            intrinsic_matrix_loaded,
            image_size,
            CV_16SC2,
            map1,
            map2
    );


    //显示经过矫正后的相机照片

    for (int i = 0;i<file_path.size();i++)
    {
        cv::Mat image;
        image = cv::imread(file_path[i]);
        if (image.empty())return;

        //重映射
        cv::remap(
                image,
                image,
                map1,
                map2,
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar()
        );

        cv::resize(image,image,cv::Size(),image_zoom,image_zoom);

        cv::imshow("Undistorted", image);
        if ((cv::waitKey(0) & 255) == 27)break;
    }
}