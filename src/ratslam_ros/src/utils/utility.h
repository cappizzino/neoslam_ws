#include <ros/ros.h>

class ParamServer
{
public:
    
    ros::NodeHandle nh;

    bool test;

    ParamServer()
    {
        nh.param<bool>("test", test, true);
        nh.setParam("value", 10);
    }
};