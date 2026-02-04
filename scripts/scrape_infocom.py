import requests
from bs4 import BeautifulSoup
import time
import os

def scrape_infocom(pages=5, output_path="data/sample_system_names.txt"):
    """
    Thu thập tên công ty từ trang danh sách của infocom.vn
    """
    base_url = "https://infocom.vn/trang-"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    new_companies = []
    
    print(f"Bắt đầu thu thập dữ liệu từ infocom.vn ({pages} trang)...")
    
    for page in range(1, pages + 1):
        url = f"{base_url}{page}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Lỗi khi truy cập trang {page}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            # Tìm tất cả các thẻ h2 chứa tên công ty dựa theo cấu trúc HTML của infocom
            headers_h2 = soup.find_all('h2')
            
            for h2 in headers_h2:
                a_tag = h2.find('a')
                if a_tag:
                    name = a_tag.text.strip()
                    # Loại bỏ số thứ tự ở đầu nếu có (ví dụ: "1 CÔNG TY..." hoặc "1000CÔNG TY...")
                    import re
                    name = re.sub(r'^\d+\s*', '', name)
                    
                    if name and len(name) > 5:
                        new_companies.append(name)
            
            print(f" Đã lấy được {len(new_companies)} công ty...")
            time.sleep(1) # Tránh bị block
            
        except Exception as e:
            print(f"Lỗi tại trang {page}: {e}")

    if not new_companies:
        print("Không tìm thấy dữ liệu mới.")
        return

    # Đọc dữ liệu cũ để tránh trùng lặp
    existing_names = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_names = {line.strip() for line in f if line.strip()}

    # Hợp nhất và lưu lại
    all_names = list(existing_names.union(set(new_companies)))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in sorted(all_names):
            f.write(f"{name}\n")
            
    print(f"\nHoàn thành! Đã cập nhật {output_path}")
    print(f"Tổng số công ty hiện có: {len(all_names)}")

if __name__ == "__main__":
    scrape_infocom(pages=100)
