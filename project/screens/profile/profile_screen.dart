import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart'; // Firebase Firestore
import 'package:krua_pa_ree/screens/register/register_screen.dart';
import '../home/home_screens.dart';

class ProfileScreen extends StatefulWidget {
  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final nameController = TextEditingController();
  final surnameController = TextEditingController();
  final phoneController = TextEditingController();
  final otherAddressController =
      TextEditingController(); // เพิ่ม Controller สำหรับ "อื่นๆ"
  String? selectedResort; // Variable to store the selected resort

  final _formKey = GlobalKey<FormState>();

  // List of resort names to display in the dropdown
  final List<String> resorts = [
    'หนานมดเเดง',
    'ลงุทิน รีสอร์ท',
    'ธาราริน รีสอร์ท',
    'ล่องเเก่งวังไม้ไผ่',
    'คุณเสือ แคมป์ปิ้ง',
    'อื่นๆ', // เพิ่มตัวเลือก "อื่นๆ"
  ];

  // Function to fetch user profile from Firebase Firestore
  Future<void> _loadUserProfile() async {
    String uid = "yourUserUID"; // Replace with actual user UID

    try {
      DocumentSnapshot userProfile =
          await FirebaseFirestore.instance.collection('users').doc(uid).get();

      if (userProfile.exists) {
        var data = userProfile.data() as Map<String, dynamic>;

        setState(() {
          nameController.text = data['name'] ?? '';
          surnameController.text = data['surname'] ?? '';
          phoneController.text = data['phone'] ?? '';
          selectedResort = data['address'] ?? '';
        });
      } else {
        setState(() {
          nameController.text = '';
          surnameController.text = '';
          phoneController.text = '';
          selectedResort = null;
        });
      }
    } catch (e) {
      print("Error loading user profile: $e");
    }
  }

  void saveProfile() async {
    if (_formKey.currentState!.validate()) {
      final address = selectedResort == 'อื่นๆ'
          ? otherAddressController.text // ใช้ค่าที่ผู้ใช้กรอก
          : selectedResort; // ใช้ค่าที่เลือกจาก Dropdown

      await updateUserProfile(
        name: nameController.text,
        surname: surnameController.text,
        address: address ?? '',
        phone: phoneController.text,
      );

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomeScreen()),
      );
    }
  }

  @override
  void initState() {
    super.initState();
    _loadUserProfile(); // Load user profile data when the screen is loaded
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(70), // กำหนดความสูงของ AppBar
        child: Container(
          decoration: BoxDecoration(
            color: Colors.orange, // สีของ AppBar
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(20), // ทำมุมมนด้านล่างซ้าย
              bottomRight: Radius.circular(20), // ทำมุมมนด้านล่างขวา
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.1),
                spreadRadius: 5,
                blurRadius: 10,
                offset: const Offset(0, 3), // เงาใต้ AppBar
              ),
            ],
          ),
          child: SafeArea(
            child: Center(
              child: Text(
                "โปรไฟล์ของคุณ", // ชื่อ AppBar
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
      ),
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Color.fromARGB(255, 252, 220, 179)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 35),
                  Center(
                    // ใช้ Center เพื่อจัดกึ่งกลาง
                    child: Image.asset(
                      'assets/images/krua pa ree.png',
                      height: 150,
                    ),
                  ),
                  _inputField("ชื่อ", nameController),
                  const SizedBox(height: 10),
                  _inputField("นามสกุล", surnameController),
                  const SizedBox(height: 10),
                  _dropdownField("ที่อยู่", selectedResort),
                  const SizedBox(height: 10),
                  _inputField("เบอร์โทรศัพท์", phoneController),
                  const SizedBox(height: 20),
                  Center(
                    child: ElevatedButton(
                      onPressed: saveProfile,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        minimumSize: const Size(double.infinity, 45),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: const Text(
                        'บันทึกข้อมูล',
                        style: TextStyle(fontSize: 16, color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
    );
  }

  Widget _inputField(String label, TextEditingController controller,
      {bool obscureText = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 5),
        TextFormField(
          controller: controller,
          obscureText: obscureText,
          decoration: InputDecoration(
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            contentPadding:
                const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
            filled: true,
            fillColor: Colors.white,
          ),
          validator: (value) {
            if (value == null || value.isEmpty) {
              return "กรุณากรอก$label";
            }
            return null;
          },
        ),
      ],
    );
  }

  Widget _dropdownField(String label, String? value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 5),
        DropdownButtonFormField<String>(
          value: value,
          hint: const Text("เลือกที่อยู่"),
          decoration: InputDecoration(
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            contentPadding:
                const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
            filled: true,
            fillColor: Colors.white,
          ),
          items: resorts.map((String resort) {
            return DropdownMenuItem<String>(
              value: resort,
              child: Text(resort),
            );
          }).toList(),
          onChanged: (String? newValue) {
            setState(() {
              selectedResort = newValue;
              if (newValue != 'อื่นๆ') {
                otherAddressController.clear(); // ลบข้อความในช่องกรอกถ้ามี
              }
            });
          },
          validator: (value) {
            if (value == null ||
                (value == 'อื่นๆ' && otherAddressController.text.isEmpty)) {
              return "กรุณาเลือกหรือกรอกที่อยู่";
            }
            return null;
          },
        ),
        if (selectedResort == 'อื่นๆ') const SizedBox(height: 10),
        if (selectedResort == 'อื่นๆ')
          TextFormField(
            controller: otherAddressController,
            decoration: InputDecoration(
              labelText: "กรอกที่อยู่",
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
              ),
              contentPadding:
                  const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
              filled: true,
              fillColor: Colors.white,
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return "กรุณากรอกที่อยู่";
              }
              return null;
            },
          ),
      ],
    );
  }
}
