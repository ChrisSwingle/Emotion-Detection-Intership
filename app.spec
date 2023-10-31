# -*- mode: python -*-

block_cipher = None


a = Analysis(['app.py'],
             pathex=['C:\\Users\\Christopher\\Documents\\ValityX\\LISA-DASH-FINAL\\LISA-DASH-FINAL'],
             binaries=[],
             datas=[('.\\venv\\Lib\\site-packages\\dash_html_components','dash_html_components'),
             ('.\\venv\\Lib\\site-packages\\dash_core_components','dash_core_components'),
             ('.\\venv\\Lib\\site-packages\\librosa','librosa'),
             ('.\\ravdess-dataset','ravdess'),
             ('.\\clf-models','clf-models'),
             ('.\\venv\\Lib\\site-packages\\pandas','pandas'),
             ('.\\venv\\Lib\\site-packages\\plotly','plotly'),
             ('.\\venv\\Lib\\site-packages\\matplotlib','matplotlib'),
             ('.\\venv\\Lib\\site-packages\\_soundfile_data\\libsndfile32bit.dll','_soundfile_data')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='app')
