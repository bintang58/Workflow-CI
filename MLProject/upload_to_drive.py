from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = 'service-account.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

def upload_file_to_drive():
    try:
        # Autentikasi ke Google Drive API
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )

        drive_service = build('drive', 'v3', credentials=credentials)

        file_path = 'models/diabetes-prediction-model.pkl'
        file_metadata = {
            'name': 'diabetes-prediction-model.pkl'
            # 'parents': ['YOUR_FOLDER_ID']  # Optional jika mau upload ke folder tertentu
        }

        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        print(f"✅ File berhasil diupload ke Google Drive dengan ID: {file.get('id')}")

    except Exception as e:
        print(f"❌ Terjadi error saat upload ke Google Drive: {e}")

if __name__ == '__main__':
    upload_file_to_drive()