@ECHO off

rem 
rem File kindly provided by Adam Hair (http://adamsccpages.blogspot.com/).
rem 
rem This example runs ordo.exe -a (replace this with an Elo rating for the 
rem reference engine)  -A"(enter name of reference engine)" -W -p (input PGN)  
rem -o rating.txt. If you want the ratings to be in a format that is easy to 
rem import to a spread sheet, use '-c' instead of '-o' and use a .csv file 
rem extension (instead of .txt). Changing the output file to rating.dat will 
rem make it compatible to Norm Pollock's embed.exe PGN utility. The following 
rem is an actual example of how to use Ordo with a batch file (a Windows file 
rem with a .bat extension that is used to send commands to programs). Make 
rem sure the batch file and Ordo are in the same folder as the input PGN (a 
rem PGN file of IPON results, named results.pgn, has been packaged with Ordo 
rem and this batch file).
 
 
ordo-win32.exe -a 2834 -A "Naum 4.2" -W -p games.pgn -o rating.txt

rem open results with notepad

notepad rating.txt

rem By double clicking on this batch file, you will cause Ordo to compute the 
rem ratings from the game results contained in results.pgn. A console window 
rem will appear (black screen, white text), and you will see Ordo's progress. 
rem The ratings will be outputted to a text file called rating.txt. Naum 4.2 
rem will be the reference engine, and its rating is set at 2834. The ratings 
rem of the other engines are relative to the rating assigned to Naum 4.2. Any 
rem engine in this or any other PGN can be used as the reference, and it can 
rem be assigned any numeric rating value. Additional information can be found 
rem in the readme.txt file that comes with Ordo. 
rem 
rem To create your own batch file, use a text editor such as NotePad, 
rem NotePad++, etc... Create a new file, type in the appropriate commands in 
rem the appropriate format (as seen above), choose save as (enter your file 
rem name).bat, and save type as 'All Files' or 'Batch' (depending on the text 
rem editor). One tip: To prevent the console window from automatically 
rem closing, type 'pause' on a second line. This can be useful if there is a 
rem problem with the batch file. It gives you a chance to see the error 
rem message.
