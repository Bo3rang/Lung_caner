#!/usr/bin/env python3
"""
Script để trích xuất text từ file PDF
Chạy script này trên máy có PyPDF2 hoặc pdfplumber
"""

import sys
import os

def extract_with_pypdf2(pdf_path):
    """Trích xuất text bằng PyPDF2"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        print("PyPDF2 không được cài đặt")
        return None

def extract_with_pdfplumber(pdf_path):
    """Trích xuất text bằng pdfplumber (tốt hơn cho tiếng Việt)"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        print("pdfplumber không được cài đặt")
        return None

def main():
    pdf_path = "Tài liệu.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} không tồn tại!")
        return
    
    print(f"Đang trích xuất text từ {pdf_path}...")
    
    # Thử pdfplumber trước (tốt hơn cho tiếng Việt)
    text = extract_with_pdfplumber(pdf_path)
    
    # Nếu không có pdfplumber, thử PyPDF2
    if text is None:
        text = extract_with_pypdf2(pdf_path)
    
    if text:
        # Lưu vào file text
        output_file = "tai_lieu_extracted.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ Đã trích xuất và lưu vào {output_file}")
        print(f"📄 Tổng số ký tự: {len(text)}")
    else:
        print("❌ Không thể trích xuất text từ PDF")

if __name__ == "__main__":
    main()