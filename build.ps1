$py_ver = "-3.9"
$py_folder = "C:/Users/Anton/AppData/Local/Programs/Python/Python39" 

Write-Output "Python at $py_folder" 

$sip_pyd = $py_folder + "\Lib\site-packages\PyQt5\sip.cp39-win_amd64.pyd;PyQt5/sip.pyd"
$qt_folder = $py_folder + "\Lib\site-packages\PyQt5\Qt5;PyQt5/Qt5"
# $cuda_dll = $py_folder + "/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll"

    # --add-binary "$cuda_dll;./onnxruntime/capi/" `

&py $py_ver -m PyInstaller app.py `
    --add-data $sip_pyd `
    --add-data $qt_folder `
    --add-binary="icon.ico;." `
    --clean --name=onnx_rt_viewer -w -F --ico icon.ico   