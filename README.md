# Face Recognition Attendance System
This project is a Face Recognition Attendance System developed using Python, OpenCV, Streamlit, and face_recognition libraries. It provides functionality to recognize faces and record attendance automatically using live video capture. Users can also manage registered members, view attendance records, and export them to CSV format.

### Features
- Live Face Recognition: Detects and recognizes registered members in real time via webcam.
- Add New Member: Register new members with multiple angle captures to improve recognition accuracy.
- View Stored Members: See all registered members with their active/inactive status.
- Attendance Records: View, download, and maintain attendance records in a tabular format.
- Delete Member: Delete a registered member securely with admin authentication.
### Technology Stack
- Python: Core programming language.
- Streamlit: Interactive user interface for real-time operations.
- OpenCV: Captures and processes webcam video frames.
- Face Recognition: Detects and matches faces based on pre-registered images.
### Project Structure
- capture_video(): Captures live video and applies face recognition to display detected members' names.
- LiveFaceRecognizer: Custom class for face recognition, member management, and attendance tracking.
- Multiple Angle Capture: Captures images of members from different angles for more accurate face detection.
- Admin Authentication: Ensures secure access to add or delete members with password protection.
- CSV Export: Provides an option to download attendance records as CSV files.
### Prerequisites
- Python 3.6+
- Streamlit
- OpenCV
- face_recognition
- Pandas
- Pillow
### Install dependencies using:

bash

pip install -r requirements.txt
### Getting Started
1. Clone the repository:

bash

git clone https://github.com/yourusername/face-recognition-attendance-system.git

2. Navigate to the project directory:
   
bash

cd face-recognition-attendance-system

3. Run the Streamlit app:
   
bash 

streamlit run app.py

4. Access the app in your browser at http://localhost:8501.

### Usage
- Run Live Face Recognition: Start recognizing faces in real time and logging attendance.
- Add New Member: Capture images for a new member from various angles, then register them with name and roll number.
- View Attendance Records: Check attendance data, and download it in CSV format.
- Delete Member: Securely delete a member from the system.
### Notes
- Ensure good lighting conditions and clear visibility of faces for better accuracy.
- Store member data and logs securely and manage sensitive information carefully.

### Author
##### Vishal Patwa











