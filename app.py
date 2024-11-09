import streamlit as st
from PIL import Image
import cv2
import numpy as np
import face_recognition as fr
import logging
import pandas as pd
import uuid
from live_face_recognizer import LiveFaceRecognizer
import time

angles = ['Center', 'Left', 'Right', 'Upper Left', 'Upper Right', 'Lower Left', 'Lower Right']

def capture_video():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        

        if not cap.isOpened():
            st.error("❌ **Cannot open webcam.**")
            return None

        video_placeholder = st.empty()
        st.write("ℹ️ **Press 'q' or Stop the app to end video stream.**")

        frame_count = 0  # Initialize frame counter

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ **Failed to capture frame.**")
                break

            if frame_count % 5 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    recognized_faces = recognizer.recognize_faces(rgb_frame)
                except Exception as e:
                    st.error(f"❌ **Face recognition error:** {e}")
                    logging.error(f"Face recognition error: {e}")
                    continue

                for face in recognized_faces:
                    (top, right, bottom, left) = face['location']
                    name = face['name']
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,0,0,), 1)

                video_placeholder.image(frame, channels="BGR", use_container_width=True)

            frame_count += 1
            time.sleep(0.03)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        st.error(f"❌ **Error during video capture:** {e}")
        logging.error(f"Error during video capture: {e}")

# ------------------------------
# 1. Set Streamlit Page Configuration
# ------------------------------
# Configure page settings
st.set_page_config(
    page_title="Face Recognition Attendance System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ------------------------------
# 3. Initialize the Recognizer

try:
    recognizer = LiveFaceRecognizer()
    st.write("✅ **Recognizer initialized successfully.**")
    logging.info("LiveFaceRecognizer initialized successfully.")
except Exception as e:
    st.error(f"❌ **Failed to initialize LiveFaceRecognizer:** {e}")
    logging.error(f"Failed to initialize LiveFaceRecognizer: {e}")
    st.stop()

# Initialize session states
if 'auth_add' not in st.session_state:
    st.session_state['auth_add'] = False
if 'auth_delete' not in st.session_state:
    st.session_state['auth_delete'] = False
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 0
if 'captured_encodings' not in st.session_state:
    st.session_state['captured_encodings'] = []
if 'add_member_in_progress' not in st.session_state:
    st.session_state['add_member_in_progress'] = False
if 'delete_in_progress' not in st.session_state:
    st.session_state['delete_in_progress'] = False
    
# ------------------------------
# 5. Sidebar Navigation with Mode Lock
# ------------------------------
# Sidebar navigation
app_mode = "Add New Member" if st.session_state['add_member_in_progress'] else (
    "Delete Member" if st.session_state['delete_in_progress'] else st.sidebar.selectbox(
        "📋 **Choose the App Mode**",
        ["Run Live Face Recognition", "Add New Member", "View Stored Members", "View Attendance Records", "Delete Member"],
        index=0
    )
)

st.sidebar.markdown("***This face recognize attendance system is made by Vishal Patwa***",unsafe_allow_html=True)
# ------------------------------
# 6. App Mode Logic
# ------------------------------

# "Add New Member" Mode
# "Add New Member" Mode
if app_mode == "Add New Member":
    st.header("➕ **Add New Member**")
    if not st.session_state['auth_add']:  # Authenticate only for adding new members
        with st.form("admin_auth_form_add"):
            password = st.text_input("🔒 **Enter admin password to add a member:**", type="password")
            submitted = st.form_submit_button("✅ **Authenticate**")

        if submitted:
            try:
                if recognizer.authenticate_admin(password):
                    st.session_state['auth_add'] = True
                    st.success("✅ **Authentication successful for adding member.**")
                    logging.info("Admin authenticated for adding member.")
                else:
                    st.error("❌ **Incorrect password.**")
                    logging.warning("Failed admin authentication for adding member.")
            except Exception as e:
                st.error(f"❌ **Authentication error:** {e}")
                logging.error(f"Authentication error for adding member: {e}")
    else:    
        # Select method for image capture
        capture_method = st.selectbox("📸 **Select Image Capture Method:**", ["Live Image Capture", "Upload Image"])
        if capture_method == "Live Image Capture":
            st.session_state['add_member_in_progress'] = True
            # Cancel button
            if st.button("🚫 **Cancel**"):
                st.session_state['add_member_in_progress'] = False
                st.session_state['current_step'] = 0
                st.session_state['captured_encodings'] = []
                st.session_state['auth_add'] = False
                st.success("Add member process canceled.")
                st.stop()
            total_steps = len(angles)
            current_step = st.session_state['current_step']

            if current_step < total_steps:
                angle = angles[current_step]
                st.subheader(f"📸 **Capture Image: {angle}**")

                try:
                    if 'cap' not in st.session_state or not st.session_state['cap'].isOpened():
                        st.session_state['cap'] = cv2.VideoCapture(0)
                        st.session_state['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        st.session_state['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    ret, frame = st.session_state['cap'].read()
                    if ret:
                        st.image(frame, channels="BGR", use_column_width=True)

                    if st.button("📷 **Capture**", key=f"capture_button_{current_step}"):
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = fr.face_locations(rgb_frame)
                        face_encodings = fr.face_encodings(rgb_frame, face_locations)

                        if len(face_encodings) == 1:
                            st.session_state['captured_encodings'].append(face_encodings[0])
                            st.success(f"✅ **Captured {angle} view successfully.**")
                            st.session_state['current_step'] += 1
                        else:
                            st.warning("⚠️ **No face detected. Please try again.**")
                except Exception as e:
                    st.error(f"❌ **Error capturing image:** {e}")
                    logging.error(f"Error capturing image at {angle} view: {e}")

                progress = (current_step / total_steps) * 100
                st.progress(progress/100)
                st.write(f"📊 **Progress:** {progress:.2f}%")

            else:
                st.success("✅ **All required images have been captured.**")
                # Collect both name and roll number
                member_name = st.text_input("📝 **Enter the name of the new member:**")
                roll_number = st.text_input("🆔 **Enter the roll number of the new member:**")

                if st.button("🚀 **Register Member**"):
                        # Ensure the session state for add_member_in_progress is updated after capturing images
                        st.session_state['add_member_in_progress'] = False
                        name = member_name.strip().upper()
                        roll_no = roll_number.strip()
                        
                        # Validating inputs
                        if name == "":
                            st.error("❌ **Member name cannot be empty.**")
                        elif roll_no == "":
                            st.error("❌ **Roll number cannot be empty.**")
                        elif name in recognizer.get_all_members():
                            st.error("❌ **This name is already registered.**")
                        else:
                            try:
                                # Register the new member with name, roll number, and captured encodings
                                recognizer.add_new_member(name, roll_no, st.session_state['captured_encodings'])
                                st.success(f"✅ **Member '{name}' with Roll Number '{roll_no}' registered successfully.**")
                                logging.info(f"New member '{name}' with Roll Number '{roll_no}' registered successfully.")
                            except Exception as e:
                                st.error(f"❌ **Failed to register member:** {e}")
                                logging.error(f"Failed to register member: {e}")
        elif capture_method == "Upload Image":
            st.session_state['add_member_in_progress'] = True
            # Cancel button
            if st.button("🚫 **Cancel**"):
                st.session_state['add_member_in_progress'] = False
                st.session_state['current_step'] = 0
                st.session_state['captured_encodings'] = []
                st.session_state['auth_add'] = False
                st.success("Add member process canceled.")
                st.stop()
            uploaded_file = st.file_uploader("📤 **Upload Image**", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Read the uploaded image
                image = Image.open(uploaded_file)
                rgb_image = np.array(image)
                face_encodings = fr.face_encodings(rgb_image)

                if len(face_encodings) == 1:
                    st.success("✅ **All required images have been captured.**")
                    # Collect both name and roll number
                    member_name = st.text_input("📝 **Enter the name of the new member:**")
                    roll_number = st.text_input("🆔 **Enter the roll number of the new member:**")

                    if st.button("🚀 **Register Member**"):
                            # Ensure the session state for add_member_in_progress is updated after capturing images
                            st.session_state['add_member_in_progress'] = False
                            name = member_name.strip().upper()
                            roll_no = roll_number.strip()
                            
                            # Validating inputs
                            if name == "":
                                st.error("❌ **Member name cannot be empty.**")
                            elif roll_no == "":
                                st.error("❌ **Roll number cannot be empty.**")
                            elif name in recognizer.get_all_members():
                                st.error("❌ **This name is already registered.**")
                            else:
                                try:
                                    # Register the new member with name, roll number, and captured encodings
                                    recognizer.add_new_member(name, roll_no, face_encodings)
                                    st.success(f"✅ **Member '{name}' with Roll Number '{roll_no}' registered successfully.**")
                                    logging.info(f"New member '{name}' with Roll Number '{roll_no}' registered successfully.")
                                except Exception as e:
                                    st.error(f"❌ **Failed to register member:** {e}")
                                    logging.error(f"Failed to register member: {e}")
                   

# "Run Live Face Recognition" Mode
elif app_mode == "Run Live Face Recognition":
    st.header("🔍 **Live Face Recognition**")
    capture_video()

elif app_mode == "View Stored Members":
    st.header("👥 **Registered Members**")
    try:
        members = recognizer.get_all_members()
        if members:
            st.write("**Registered Members:**")
            for member, active in members.items():
                status = "Active" if active else "Inactive"
                st.write(f"- {member} ({status})")
        else:
            st.write("❌ **No members found.**")
        logging.info("Displayed stored members.")
    except Exception as e:
        st.error(f"❌ **Error fetching members:** {e}")
        logging.error(f"Error fetching members: {e}")

elif app_mode == "View Attendance Records":
    st.header("📋 **Attendance Records**")
    try:
        attendance_records = recognizer.get_attendance_records()
        if attendance_records:
            df = pd.DataFrame(attendance_records)
            st.dataframe(df)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Attendance CSV", csv, "attendance_records.csv", "text/csv")
        else:
            st.write("❌ **No attendance records found.**")
        logging.info("Displayed attendance records.")
    except Exception as e:
        st.error(f"❌ **Error fetching attendance records:** {e}")
        logging.error(f"Error fetching attendance records: {e}")

elif app_mode == "Delete Member":
    st.header("❌ **Delete Member**")
    if not st.session_state['auth_delete']:  # Authenticate only for deleting members
        with st.form("admin_auth_form_delete"):
            password = st.text_input("🔒 **Enter admin password to delete a member:**", type="password")
            submitted = st.form_submit_button("✅ **Authenticate**")

        if submitted:
            try:
                if recognizer.authenticate_admin(password):
                    st.session_state['auth_delete'] = True
                    st.session_state['delete_in_progress'] = True
                    st.success("✅ **Authentication successful for deleting member.**")
                    logging.info("Admin authenticated for deleting member.")
                else:
                    st.error("❌ **Incorrect password.**")
                    logging.warning("Failed admin authentication for deleting member.")
            except Exception as e:
                st.error(f"❌ **Authentication error:** {e}")
                logging.error(f"Authentication error for deleting member: {e}")
    else:   
        try:
            # Cancel button
            if st.button("🚫 **Cancel**"):
                st.session_state['auth_delete'] = False
                st.session_state['delete_in_progress'] = False
                st.success("Delete member process canceled.")
                st.stop()
            members = recognizer.get_all_members()
            if members:
                member_to_delete = st.selectbox("Select a member to delete:", members.keys())
                if st.button("🗑️ **Delete Member**"):
                    try:
                        if recognizer.delete_member(member_to_delete):
                            st.success(f"✅ **Member '{member_to_delete}' deleted successfully.**")
                            logging.info(f"Deleted member: {member_to_delete}")

                        else:
                            st.error(f"❌ **Failed to delete member: {member_to_delete}.**")
                            logging.warning(f"Failed to delete non-existing member: {member_to_delete}.")
                    except Exception as e:
                        st.error(f"❌ **Error deleting member:** {e}")
                        logging.error(f"Error deleting member '{member_to_delete}': {e}")
            else:
                st.write("❌ **No members found.**")
        except Exception as e:
            st.error(f"❌ **Error fetching members for deletion:** {e}")
            logging.error(f"Error fetching members for deletion: {e}")