contains(DEFINES, websocket) {
HEADERS += $$PWD/webclient.h
HEADERS += $$PWD/webserver.h

SOURCES += $$PWD/webclient.cpp
SOURCES += $$PWD/webserver.cpp
}

HEADERS += \
    $$PWD/appconfig.h \
    $$PWD/appdata.h \
    $$PWD/databasemanager.h \
    $$PWD/qthelper.h \
    $$PWD/qthelperdata.h \
    $$PWD/tcpclient.h \
    $$PWD/tcpserver.h

SOURCES += \
    $$PWD/appconfig.cpp \
    $$PWD/appdata.cpp \
    $$PWD/databasemanager.cpp \
    $$PWD/qthelper.cpp \
    $$PWD/qthelperdata.cpp \
    $$PWD/tcpclient.cpp \
    $$PWD/tcpserver.cpp

# api.pri
INCLUDEPATH += $$PWD  # 添加自身目录到头文件搜索路径
DEPENDPATH += $$PWD
