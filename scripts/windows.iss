# Inno Setup script for StudyGlance overlay (basic example)
# Requires Inno Setup (ISCC) installed.

[Setup]
AppName=StudyGlance
AppVersion=0.1.0
DefaultDirName={autopf}\StudyGlance
DefaultGroupName=StudyGlance
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=StudyGlance-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\StudyGlance.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\StudyGlance"; Filename: "{app}\StudyGlance.exe"
Name: "{autodesktop}\StudyGlance"; Filename: "{app}\StudyGlance.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

