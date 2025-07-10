#include "frmmain.h"
#include "ui_frmmain.h"
#include "qthelper.h"

#include "frmtcpclient.h"
#include "frmtcpserver.h"
#ifdef websocket
#include "frmwebclient.h"
#include "frmwebserver.h"
#endif

frmMain::frmMain(QWidget *parent) : QWidget(parent), ui(new Ui::frmMain)
{
    ui->setupUi(this);

}

frmMain::~frmMain()
{
    delete ui;
}

void frmMain::showMainInterface()
{
    this->initForm();
    this->initConfig();
    this->show();
}

void frmMain::initForm()
{
    ui->tabWidget->addTab(new frmTcpClient, "TCP客户端");
    ui->tabWidget->addTab(new frmTcpServer, "TCP服务端");

#ifdef websocket
    ui->tabWidget->addTab(new frmWebClient, "WEB客户端");
    ui->tabWidget->addTab(new frmWebServer, "WEB服务端");
#endif
#ifdef Q_OS_WASM
    AppConfig::CurrentIndex = 4;
#endif
}

void frmMain::initConfig()
{
    ui->tabWidget->setCurrentIndex(AppConfig::CurrentIndex);
    connect(ui->tabWidget, SIGNAL(currentChanged(int)), this, SLOT(saveConfig()));
}

void frmMain::saveConfig()
{
    AppConfig::CurrentIndex = ui->tabWidget->currentIndex();
    AppConfig::writeConfig();
}
