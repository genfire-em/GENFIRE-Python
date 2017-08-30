#-------------------------------------------------
#
# Project created by QtCreator 2016-02-09T15:04:18
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 5): QT += widgets

TARGET = GENFIRE

TEMPLATE = app


SOURCES += GENFIRE_MainWindow.cpp

HEADERS  += GENFIRE_MainWindow.h

FORMS    += GENFIRE_MainWindow.ui \
            ProjectionCalculator_MainWindow.ui \
            VolumeSlicer_MainWindow.ui \
    CalculateProjectionSeries_Dialog.ui \
    calculateprojectionseries_dialog.ui

RESOURCES   += GENFIRE.qrc
