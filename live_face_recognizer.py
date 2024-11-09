# live_face_recognizer.py
import face_recognition as fr
import pickle
import os
import numpy as np
import csv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class LiveFaceRecognizer:
    def __init__(self, data_file='face_data.pkl', attendance_file='attendance.csv', admin_password='admin123'):
        self.members = {}  # Dictionary to store name: [encodings, status]
        self.data_file = data_file
        self.attendance_file = attendance_file
        self.admin_password = admin_password
        self.load_data()
        self.initialize_attendance_file()
        logging.info("LiveFaceRecognizer initialized.")

    def authenticate_admin(self, password):
        return password == self.admin_password

    def add_new_member(self, name, roll_no, face_encodings):
        if name in self.members:
            logging.warning(f"Attempted to add existing member: {name}")
            return False  # Name already exists
        else:
            # Store name, roll number, and encodings in a dictionary
            self.members[name] = {
                'roll_no': roll_no,
                'encodings': face_encodings,
                'active': True
            }
            self.save_data()
            logging.info(f"Added new member: {name} (Roll No: {roll_no}) with {len(face_encodings)} encodings.")
            return True

    def delete_member(self, name):
        if name in self.members:
            del self.members[name]
            self.save_data()
            print("GO")
            logging.info(f"Deleted member: {name}")
            return True
        else:
            logging.warning(f"Attempted to delete non-existing member: {name}")
            return False  # Member not found

    def recognize_faces(self, image):
        recognized_faces = []
        face_locations = fr.face_locations(image, model='hog')
        face_encodings = fr.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            best_match_distance = float('inf')

            for member_name, member_data in self.members.items():
                if not member_data['active']:  # Skip inactive members
                    continue
                matches = fr.compare_faces(member_data['encodings'], face_encoding, tolerance=0.6)
                face_distances = fr.face_distance(member_data['encodings'], face_encoding)

                if len(face_distances) == 0:
                    continue
                min_distance = min(face_distances)
                if min_distance < best_match_distance and matches[np.argmin(face_distances)]:
                    best_match_distance = min_distance
                    name = member_name

            if name != "Unknown":
                self.mark_attendance(name)

            recognized_faces.append({
                'location': (top, right, bottom, left),
                'name': name
            })

        logging.info(f"Recognized faces: {[face['name'] for face in recognized_faces]}")
        return recognized_faces

    def mark_attendance(self, name):
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")

        already_present = False
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                roll_no = self.members[name]['roll_no'] if name in self.members else None
                for row in reader:
                    if row['Roll No'] == self.members[name]['roll_no'] and row['Date'] == date_string:
                        already_present = True
                        break

        if not already_present:
            roll_no = self.members[name]['roll_no']  # Retrieve roll number
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([roll_no, name, date_string, time_string])
            logging.info(f"Attendance marked for {name} (Roll No: {roll_no}) at {time_string} on {date_string}.")
        
    def initialize_attendance_file(self):
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Roll No', 'Name', 'Date', 'Time'])  # Added Roll No
            logging.info(f"Attendance file '{self.attendance_file}' created.")
        else:
            logging.info(f"Attendance file '{self.attendance_file}' found.")

    def save_data(self):
        logging.debug("Saving data...")
        with open(self.data_file, 'wb') as f:
            pickle.dump({'members': self.members}, f)
        logging.info("Face data saved successfully.")

    def load_data(self):
        if os.path.exists(self.data_file):
            if os.path.getsize(self.data_file) == 0:
                logging.warning("The data file is empty. Starting fresh.")
                self.members = {}
                return
            
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    if not data or 'members' not in data:
                        raise ValueError("Invalid data format in pickle file.")
                    self.members = data['members']
                    logging.info(f"Loaded {len(self.members)} member(s) from storage.")
            except (pickle.UnpicklingError, ValueError) as e:
                logging.error(f"Failed to load data due to corruption or invalid format: {e}")
                self.members = {}
        else:
            logging.info("No existing face data found. Starting fresh.")
            
    def get_all_members(self):
        return {name: data['active'] for name, data in self.members.items()}

    def get_attendance_records(self):
        if not os.path.exists(self.attendance_file):
            return []
        with open(self.attendance_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def get_data(self):
        return self.members