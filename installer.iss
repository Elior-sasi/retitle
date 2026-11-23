; Retitle installer (Inno Setup 6). Per-user install, no admin rights needed.
#ifndef AppVer
  #define AppVer "0"
#endif

[Setup]
AppId={{6E2F4D1B-7C0B-4E57-9F1F-3A5C2B7D9E10}
AppName=Retitle
AppVersion={#AppVer}
AppPublisher=Elior
DefaultDirName={localappdata}\Programs\Retitle
DefaultGroupName=Retitle
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=Retitle-Setup
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\Retitle.exe
SetupLogging=yes
ChangesAssociations=yes
SetupIconFile=assets\icon.ico
WizardSmallImageFile=assets\icon_wizard.bmp

[Languages]
Name: "he"; MessagesFile: "compiler:Languages\Hebrew.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "dist\Retitle\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; .retitle project files open in Retitle (double-click), with the app icon
Root: HKA; Subkey: "Software\Classes\.retitle"; ValueType: string; ValueName: ""; ValueData: "Retitle.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\Retitle.Project"; ValueType: string; ValueName: ""; ValueData: "Retitle project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\Retitle.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Retitle.exe,0"
Root: HKA; Subkey: "Software\Classes\Retitle.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Retitle.exe"" ""%1"""

[Icons]
Name: "{group}\Retitle"; Filename: "{app}\Retitle.exe"
Name: "{group}\{cm:UninstallProgram,Retitle}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Retitle"; Filename: "{app}\Retitle.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Retitle.exe"; Description: "{cm:LaunchProgram,Retitle}"; Flags: nowait postinstall skipifsilent
; after a silent self-update the app is started again automatically
Filename: "{app}\Retitle.exe"; Flags: nowait; Check: WizardSilent
