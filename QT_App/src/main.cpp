#include "frmmain.h"
#include "qthelper.h"
#include "frmLogin.h"
#include <QApplication>
#include <QMessageBox>

int main(int argc, char *argv[])
{
    QtHelper::initMain();
    QApplication a(argc, argv);
    a.setWindowIcon(QIcon(":/main.ico"));


    // 初始化配置（必须在登录前读取，因为登录可能依赖配置）
    QtHelper::initAll();
    AppConfig::ConfigFile = QString("%1/%2.ini").arg(QtHelper::appPath()).arg(QtHelper::appName());
    AppConfig::readConfig();  // 这里会读取CurrentIndex

    // 初始化数据
    AppData::Intervals << "1" << "10" << "20" << "50" << "100" << "200" << "300" << "500" << "1000" << "1500" << "2000" << "3000" << "5000" << "10000";
    AppData::readSendData();
    AppData::readDeviceData();

    // 显示登录窗口
    frmLogin loginDialog;
    loginDialog.setWindowTitle("巡检小车-用户登录");
    QtHelper::setFormInCenter(&loginDialog);

    if (loginDialog.exec() != QDialog::Accepted) {
        return 0;  // 登录取消
    }

    // 创建主窗口并初始化标签页
    frmMain w;
    w.setWindowTitle("巡检小车");
    w.resize(950, 700);

    // 关键修改：先初始化界面再显示
    w.initForm();      // 初始化标签页
    w.initConfig();    // 恢复上次的标签页索引
    QtHelper::setFormInCenter(&w);
    w.show();

    return a.exec();
}
