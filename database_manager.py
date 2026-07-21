import os


class DatabaseManager:
    def __init__(self, base_dir="database"):
        self.base_dir = base_dir
        self.data = {"YesNo": {}, "Personality": {}}
        self.load_database()

    def load_database(self):
        """Сканирует папки YesNo и Personality и загружает текст из .txt файлов."""
        for folder in ["YesNo", "Personality"]:
            folder_path = os.path.join(self.base_dir, folder)

            if not os.path.exists(folder_path):
                print(f"Warning: Folder '{folder_path}' not found.")
                continue

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    card_name = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text_content = f.read().strip()

                        cleaned_key = card_name.lower().strip()
                        self.data[folder][cleaned_key] = text_content
                    except Exception as e:
                        print(f"❌ Error reading file '{file_name}': {e}")


    def get_prediction(self, mode_idx, card_name):
        """Возвращает текст предсказания для карты."""
        if mode_idx == 1 or mode_idx == "YesNo" or mode_idx == "Yes/No":
            folder = "YesNo"
        else:
            folder = "Personality"

        cleaned_card = str(card_name).lower().strip()

        try:
            return self.data[folder][cleaned_card]
        except KeyError:
            return f"Prediction text for '{card_name}' is missing in {folder} database."