FORMS += $$PWD/frmmain.ui
FORMS += $$PWD/frmlogin.ui
FORMS += $$PWD/frmtcpclient.ui
FORMS += $$PWD/frmtcpserver.ui
FORMS +=
FORMS +=

HEADERS += $$PWD/frmmain.h
HEADERS += $$PWD/frmLogin.h
HEADERS += $$PWD/frmtcpclient.h
HEADERS += $$PWD/frmtcpserver.h
HEADERS +=
HEADERS +=

SOURCES += $$PWD/frmmain.cpp
SOURCES += $$PWD/frmLogin.cpp
SOURCES += $$PWD/frmtcpclient.cpp
SOURCES += $$PWD/frmtcpserver.cpp
SOURCES +=
SOURCES +=

contains(DEFINES, websocket) {
FORMS   += $$PWD/frmwebclient.ui
FORMS   += $$PWD/frmwebserver.ui

HEADERS += $$PWD/frmwebclient.h
HEADERS += $$PWD/frmwebserver.h

SOURCES += $$PWD/frmwebclient.cpp
SOURCES += $$PWD/frmwebserver.cpp
}
