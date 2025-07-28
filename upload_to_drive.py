from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import sys
import os

# ---- SETTINGS ----
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

def authenticate():
    """
    Authenticate and return a Google Drive service object.
    Saves token.json for reuse.
    """
    import pickle

    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"Please go to this URL:\n\n{auth_url}\n")
            code = input("Enter the authorization code: ")
            creds = flow.fetch_token(code=code)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service

def upload_file(service, local_path, folder_id):
    """
    Upload a local file to Google Drive in specified folder.
    """
    file_metadata = {
        'name': os.path.basename(local_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(local_path, resumable=True)
    uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"? Uploaded. File ID: {uploaded.get('id')}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upload_to_drive.py <local_file> <folder_id>")
        sys.exit(1)

    local_file = sys.argv[1]
    folder_id = sys.argv[2]

    service = authenticate()
    upload_file(service, local_file, folder_id)
