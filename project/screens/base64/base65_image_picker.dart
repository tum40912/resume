import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

class Base64ImagePicker1 extends StatefulWidget {
  @override
  _Base64ImagePickerState createState() => _Base64ImagePickerState();
}

class _Base64ImagePickerState extends State<Base64ImagePicker1> {
  Uint8List? _imageBytes;  // เก็บข้อมูลรูปภาพที่เลือก
  String? paymentSlip;  // ตัวแปรใหม่สำหรับเก็บข้อมูล Payment Slip

  Future<void> pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom, 
      allowedExtensions: ['jpg', 'png', 'jpeg'], // กำหนดไฟล์ที่อนุญาต
      allowMultiple: false,
    );

    if (result != null && result.files.single.bytes != null) {
      setState(() {
        _imageBytes = result.files.single.bytes;  // เก็บข้อมูลรูปภาพที่เลือก
        paymentSlip = "paymentSlip_${DateTime.now().millisecondsSinceEpoch}";  // ตั้งชื่อฟิลด์ paymentSlip ตามเวลาปัจจุบัน
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
                      Navigator.pop(context, {
                        'image': _imageBytes, // ส่งข้อมูลรูปภาพ
                        'paymentSlip': paymentSlip, // ส่งฟิลด์ paymentSlip
                      });
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
