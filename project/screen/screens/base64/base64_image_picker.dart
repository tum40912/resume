import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:file_picker/file_picker.dart';


class Base64ImagePicker extends StatefulWidget {
  @override
  _Base64ImagePickerState createState() => _Base64ImagePickerState();
}

class _Base64ImagePickerState extends State<Base64ImagePicker> {
  Uint8List? _imageBytes;

  Future<void> pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
  type: FileType.custom, 
  allowedExtensions: ['jpg', 'png', 'jpeg'], // กำหนดไฟล์ที่อนุญาต
  allowMultiple: false,
);


    if (result != null && result.files.single.bytes != null) {
      setState(() {
        _imageBytes = result.files.single.bytes; // ใช้ bytes แทน File
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("เลือกรูปภาพ")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _imageBytes == null
                ? const Text("เลือกรูปภาพ", style: TextStyle(fontSize: 18))
                : Image.memory(_imageBytes!, fit: BoxFit.cover, height: 200, width: 200), // แสดงรูปภาพ
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: pickImage,
              child: const Text("เลือกภาพจากไฟล์"),
            ),
            const SizedBox(height: 16),
            _imageBytes != null
                ? ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context, _imageBytes); // ส่งรูปกลับไปหน้า `AddItemsScreen`
                    },
                    child: const Text("ยืนยันรูปภาพ"),
                  )
                : Container(),
          ],
        ),
      ),
    );
  }
}
