@echo off
echo =====================================
echo Training Toxic Comments Classifier Model
echo =====================================
echo.
echo This will train the ML model using sample data.
echo Training may take a few minutes...
echo.

cd backend
python train_model.py

echo.
echo Training complete!
echo You can now run the application with the trained model.
echo.
pause
