; Script Inno Setup tao bo cai dat chuyen nghiep cho He Thong Giam Sat Duong Sat
#define MyAppName "He Thong Giam Sat An Toan Duong Sat"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Railway Safety Team"
#define MyAppExeName "RailwaySafety.exe"

[Setup]
; Ma GUID dinh danh ung dung
AppId={{5A8D2F1B-3C9E-4E7B-8A12-9F0D5E8A9C1D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Cai dat mac dinh vao thu muc de co toan quyen ghi video va log
DefaultDirName=C:\RailwaySafety
DisableDirPage=no
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=Setup_RailwaySafety_v1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Can quyen Administrator de cai dat Visual C++ vao he thong
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; 1. Sao chep toan bo thu muc ung dung da dong goi
Source: "dist\RailwaySafety\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 2. Bo cai Microsoft Visual C++ (tu dong xoa sau khi cai xong de tiet kiem bo nho)
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; 1. Tu dong cai dat Microsoft Visual C++ 2015-2022 o che do chay ngam (Silent)
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /passive /norestart"; Flags: waituntilterminated; StatusMsg: "Dang kiem tra va cap nhat thu vien Microsoft Visual C++..."
; 2. Tich chon mo ung dung ngay sau khi cai dat xong
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

