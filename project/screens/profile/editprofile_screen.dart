import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class EditProfileScreen extends StatefulWidget {
  final String userId;

  const EditProfileScreen({Key? key, required this.userId}) : super(key: key);

  @override
  _EditProfileScreenState createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController surnameController = TextEditingController();
  final TextEditingController phoneController = TextEditingController();
  String? selectedAddress;
  final otherAddressController =
      TextEditingController(); // เพิ่ม Controller สำหรับ "อื่นๆ"
  String? selectedResort;
  // ตัวเลือกสำหรับที่อยู่
  final List<String> addressOptions = [
    'หนานมดแดง',
    'ลุงทิน รีสอร์ท',
    'ธาราริน รีสอร์ท',
    'ล่องแก่งวังไม้ไผ่',
    'คุณเสือ แคมป์ปิ้ง',
    'อื่นๆ',
  ];

  void showSuccessDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.check_circle, size: 70, color: Colors.green),
                const SizedBox(height: 10),
                Text(
                  "อัปเดตโปรไฟล์สำเร็จ!",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 10),
                const Text(
                  "ข้อมูลของคุณได้รับการบันทึกเรียบร้อยแล้ว 🎉",
                  style: TextStyle(fontSize: 16, color: Colors.grey),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.pop(context); // ปิด popup
                    Navigator.pop(context); // กลับไปหน้าก่อนหน้า
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    padding: EdgeInsets.symmetric(horizontal: 30, vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: const Text("ตกลง",
                      style: TextStyle(fontSize: 18, color: Colors.white)),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _loadUserData() async {
    try {
      final doc = await FirebaseFirestore.instance
          .collection('Customers')
          .doc(widget.userId)
          .get();

      if (doc.exists) {
        final data = doc.data() as Map<String, dynamic>;
        nameController.text = data['name'] ?? '';
        surnameController.text = data['surname'] ?? '';
        phoneController.text = data['phone'] ?? '';
        selectedAddress = data['address'] ?? null;
      }
    } catch (e) {
      print('Error loading user data: $e');
    }
  }
  
  void showErrorDialog(BuildContext context, String message) {
  showDialog(
    context: context,
    builder: (context) {
      return AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(15), // ✅ ขอบโค้งมน
        ),
        title: Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.red, size: 28), // ✅ ไอคอนเตือน
            SizedBox(width: 8),
            Text(
              "แจ้งเตือน",
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        content: Text(
          message,
          style: TextStyle(fontSize: 16),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context), // ปิด popup
            child: Text(
              "ตกลง",
              style: TextStyle(color: Colors.blue, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      );
    },
  );
}


  Future<void> _saveUserData() async {
  try {
    // ตรวจสอบว่าทุกช่องต้องไม่ว่าง
    if (nameController.text.trim().isEmpty ||
        surnameController.text.trim().isEmpty ||
        phoneController.text.trim().isEmpty ||
        (selectedAddress == 'อื่นๆ' && otherAddressController.text.trim().isEmpty)) {
     showErrorDialog(context, 'กรุณากรอกข้อมูลให้ครบถ้วน');

      return;
    }

    // ตรวจสอบว่าเบอร์โทรขึ้นต้นด้วย 0 และมีความยาวที่เหมาะสม
    String phoneNumber = phoneController.text.trim();
    if (!RegExp(r'^0[0-9]{8,9}$').hasMatch(phoneNumber)) {
      showErrorDialog(context, 'กรุณากรอกเบอร์โทรให้ถูกต้อง');

      return;
    }

    String finalAddress = selectedAddress == 'อื่นๆ'
        ? otherAddressController.text.trim()
        : selectedAddress ?? '';

    await FirebaseFirestore.instance
        .collection('Customers')
        .doc(widget.userId)
        .update({
      'name': nameController.text.trim(),
      'surname': surnameController.text.trim(),
      'phone': phoneNumber,
      'address': finalAddress,
    });

    showSuccessDialog(context);
  } catch (e) {
    print('Error saving user data: $e');
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('เกิดข้อผิดพลาดในการอัปเดตข้อมูล')),
    );
  }
}


  @override
  void initState() {
    super.initState();
    _loadUserData();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60), // กำหนดความสูงของ AppBar
        child: ClipRRect(
          borderRadius: const BorderRadius.only(
            bottomLeft: Radius.circular(20), // ขอบโค้งมนด้านซ้ายล่าง
            bottomRight: Radius.circular(20), // ขอบโค้งมนด้านขวาล่าง
          ),
          child: AppBar(
            flexibleSpace: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(0.5), // สีส้มไล่เฉด
                    Colors.orangeAccent,
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
            title: const Text(
              "แก้ไขข้อมูลส่วนตัว",
              style: TextStyle(
                fontFamily: "assets/fonts/ChakraPetch-Bold.ttf",
                color: Color.fromARGB(255, 0, 0, 0),
                fontWeight: FontWeight.bold,
              ),
            ),
            centerTitle: true, // จัดกึ่งกลางข้อความ
            elevation: 5, // เพิ่มเงา
          ),
        ),
      ),
      body: Container(
        width: double.infinity,
        height: double.infinity, // ทำให้ Container ขยายเต็มจอ
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Color.fromARGB(255, 252, 220, 179)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                const SizedBox(height: 16),
                Image.asset(
                  'assets/images/krua pa ree.png',
                  height: 155, // หรือปรับเพิ่มตามความเหมาะสม
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(
                    labelText: "ชื่อ",
                    filled: true,
                    fillColor: Color.fromARGB(255, 255, 227, 185),
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: surnameController,
                  decoration: const InputDecoration(
                    labelText: "นามสกุล",
                    filled: true,
                    fillColor: Color.fromARGB(255, 255, 227, 185),
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<String>(
                  value: selectedAddress,
                  decoration: const InputDecoration(
                    labelText: "ที่อยู่",
                    filled: true,
                    fillColor: Color.fromARGB(255, 255, 227, 185),
                    border: OutlineInputBorder(),
                  ),
                  items: addressOptions
                      .map((address) => DropdownMenuItem(
                            value: address,
                            child: Text(address),
                          ))
                      .toList(),
                  onChanged: (value) {
                    setState(() {
                      selectedAddress = value;
                    });
                  },
                ),
                if (selectedAddress == 'อื่นๆ')
                  Padding(
                    padding: const EdgeInsets.only(top: 10),
                    child: TextField(
                      controller: otherAddressController,
                      decoration: const InputDecoration(
                        labelText: "กรอกที่อยู่",
                        filled: true,
                        fillColor: Color.fromARGB(255, 255, 227, 185),
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                const SizedBox(height: 16),
                TextField(
                  controller: phoneController,
                  keyboardType: TextInputType.phone,
                  decoration: const InputDecoration(
                    labelText: "เบอร์โทรศัพท์",
                    filled: true,
                    fillColor: Color.fromARGB(255, 255, 227, 185),
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 32),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () async {
                      await _saveUserData(); // บันทึกข้อมูลใน Firestore
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: const Text(
                      "บันทึกข้อมูล",
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
