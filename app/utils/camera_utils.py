import sys
import cv2

def get_system_cameras():
    """
    Liệt kê danh sách các camera DirectShow trên hệ thống Windows theo đúng thứ tự index của OpenCV.
    Trả về danh sách dict: [{'index': int, 'name': str, 'is_obs': bool}]
    """
    cameras = []
    
    if sys.platform.startswith('win'):
        try:
            import ctypes
            from ctypes import wintypes, Structure, byref, c_void_p, cast, POINTER

            ole32 = ctypes.windll.ole32
            ole32.CoInitialize(None)

            class GUID(Structure):
                _fields_ = [
                    ('Data1', wintypes.DWORD),
                    ('Data2', wintypes.WORD),
                    ('Data3', wintypes.WORD),
                    ('Data4', wintypes.BYTE * 8)
                ]
                def __init__(self, l, w1, w2, b):
                    self.Data1 = l
                    self.Data2 = w1
                    self.Data3 = w2
                    self.Data4 = (wintypes.BYTE * 8)(*b)

            CLSID_SystemDeviceEnum = GUID(0x62be5d10, 0x60eb, 0x11d0, (0xbd, 0x3b, 0x00, 0xa0, 0xc9, 0x11, 0xce, 0x86))
            CLSID_VideoInputDeviceCategory = GUID(0x860bb310, 0x5d01, 0x11d0, (0xbd, 0x3b, 0x00, 0xa0, 0xc9, 0x11, 0xce, 0x86))
            IID_ICreateDevEnum = GUID(0x29840822, 0x5b84, 0x11d0, (0xbd, 0x3b, 0x00, 0xa0, 0xc9, 0x11, 0xce, 0x86))
            IID_IPropertyBag = GUID(0x55272a00, 0x42cb, 0x11ce, (0x81, 0x35, 0x00, 0xaa, 0x00, 0x4b, 0xb8, 0x51))

            class ICreateDevEnumVtbl(Structure):
                _fields_ = [
                    ('QueryInterface', c_void_p),
                    ('AddRef', c_void_p),
                    ('Release', ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)),
                    ('CreateClassEnumerator', ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, POINTER(GUID), POINTER(c_void_p), wintypes.DWORD))
                ]

            class ICreateDevEnum(Structure):
                _fields_ = [('lpVtbl', POINTER(ICreateDevEnumVtbl))]

            class IEnumMonikerVtbl(Structure):
                _fields_ = [
                    ('QueryInterface', c_void_p),
                    ('AddRef', c_void_p),
                    ('Release', ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)),
                    ('Next', ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, wintypes.ULONG, POINTER(c_void_p), POINTER(wintypes.ULONG))),
                    ('Skip', c_void_p),
                    ('Reset', c_void_p),
                    ('Clone', c_void_p)
                ]

            class IEnumMoniker(Structure):
                _fields_ = [('lpVtbl', POINTER(IEnumMonikerVtbl))]

            class IMonikerVtbl(Structure):
                _fields_ = [
                    ('QueryInterface', c_void_p),
                    ('AddRef', c_void_p),
                    ('Release', ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)),
                    ('GetClassID', c_void_p),
                    ('IsDirty', c_void_p),
                    ('Load', c_void_p),
                    ('Save', c_void_p),
                    ('GetSizeMax', c_void_p),
                    ('BindToObject', c_void_p),
                    ('BindToStorage', ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p)))
                ]

            class IMoniker(Structure):
                _fields_ = [('lpVtbl', POINTER(IMonikerVtbl))]

            class VARIANT(Structure):
                _fields_ = [
                    ('vt', wintypes.WORD),
                    ('wReserved1', wintypes.WORD),
                    ('wReserved2', wintypes.WORD),
                    ('wReserved3', wintypes.WORD),
                    ('bstrVal', c_void_p),
                    ('padding', wintypes.BYTE * 8)
                ]

            class IPropertyBagVtbl(Structure):
                _fields_ = [
                    ('QueryInterface', c_void_p),
                    ('AddRef', c_void_p),
                    ('Release', ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)),
                    ('Read', ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, wintypes.LPCWSTR, POINTER(VARIANT), c_void_p)),
                    ('Write', c_void_p)
                ]

            class IPropertyBag(Structure):
                _fields_ = [('lpVtbl', POINTER(IPropertyBagVtbl))]

            pDevEnum = c_void_p()
            hr = ole32.CoCreateInstance(byref(CLSID_SystemDeviceEnum), None, 1, byref(IID_ICreateDevEnum), byref(pDevEnum))
            if hr == 0 and pDevEnum:
                devEnum = cast(pDevEnum, POINTER(ICreateDevEnum))
                pEnumMoniker = c_void_p()
                hr_enum = devEnum.contents.lpVtbl.contents.CreateClassEnumerator(pDevEnum, byref(CLSID_VideoInputDeviceCategory), byref(pEnumMoniker), 0)
                if hr_enum == 0 and pEnumMoniker:
                    enumMoniker = cast(pEnumMoniker, POINTER(IEnumMoniker))
                    pMoniker = c_void_p()
                    fetched = wintypes.ULONG()
                    idx = 0
                    while enumMoniker.contents.lpVtbl.contents.Next(pEnumMoniker, 1, byref(pMoniker), byref(fetched)) == 0:
                        moniker = cast(pMoniker, POINTER(IMoniker))
                        pPropBag = c_void_p()
                        hr_bag = moniker.contents.lpVtbl.contents.BindToStorage(pMoniker, None, None, byref(IID_IPropertyBag), byref(pPropBag))
                        name = f"Camera {idx}"
                        if hr_bag == 0 and pPropBag:
                            propBag = cast(pPropBag, POINTER(IPropertyBag))
                            var = VARIANT()
                            var.vt = 8 # VT_BSTR
                            if propBag.contents.lpVtbl.contents.Read(pPropBag, 'FriendlyName', byref(var), None) == 0:
                                name = ctypes.wstring_at(var.bstrVal)
                            propBag.contents.lpVtbl.contents.Release(pPropBag)
                        moniker.contents.lpVtbl.contents.Release(pMoniker)
                        cameras.append({
                            'index': idx,
                            'name': name,
                            'is_obs': 'obs' in name.lower()
                        })
                        idx += 1
                    enumMoniker.contents.lpVtbl.contents.Release(pEnumMoniker)
                devEnum.contents.lpVtbl.contents.Release(pDevEnum)
        except Exception as e:
            print(f"[!] Lỗi khi quét DirectShow camera: {e}")

    # Dự phòng nếu không quét được qua COM
    if not cameras:
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({
                    'index': i,
                    'name': f"Camera {i}",
                    'is_obs': False
                })
                cap.release()

    return cameras

def find_obs_camera_index(configured_val="auto", cameras=None):
    """
    Xác định index của OBS Virtual Camera.
    1. Nếu người dùng cấu hình một số nguyên (ví dụ: 0, 1...), ưu tiên trả về số đó.
    2. Nếu cấu hình là 'auto' hoặc rỗng, tự động tìm camera có tên chứa 'OBS'.
    3. Nếu không tìm thấy chữ 'OBS', ưu tiên index 1 nếu có nhiều cam, ngược lại index 0.
    """
    if configured_val is not None:
        val_str = str(configured_val).strip().lower()
        if val_str not in ("auto", "", "none"):
            try:
                idx = int(val_str)
                if idx >= 0:
                    return idx
            except ValueError:
                pass

    if cameras is None:
        cameras = get_system_cameras()
    # Tìm camera có 'obs' trong tên
    for cam in cameras:
        if cam.get('is_obs'):
            return cam['index']

    # Nếu không thấy OBS rõ ràng, nếu có >= 2 camera, chọn camera thứ 2 (index 1) vì thường index 0 là webcam tích hợp
    if len(cameras) >= 2:
        return 1
    elif len(cameras) == 1:
        return cameras[0]['index']

    return 0
