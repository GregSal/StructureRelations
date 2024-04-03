rem ########## VS Code Launch ############
rem set WORKSPACE_FOLDER="%HOMEPATH%\OneDrive - Queen's University\Python\Projects\sectionary package"
set WORKSPACE_FOLDER="D:\OneDrive - Queen's University\Python\Projects\StructureRelations"
set WORKSPACE_FILE="%WORKSPACE_FOLDER%\StructureRelations.code-workspace"

CALL D:\anaconda3\Scripts\activate.bat D:\anaconda3
CALL conda activate StructureRelations
Cd "%WORKSPACE_FOLDER%"
D:
CALL code StructureRelations.code-workspace
EXIT

