import 'dart:convert';
import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:krua_pa_ree/screens/additems/additems_screens.dart';
import 'package:krua_pa_ree/screens/esititemp/edititem_screen.dart';

class FoodMenumenuScreen extends StatefulWidget {
  const FoodMenumenuScreen({Key? key}) : super(key: key);

  @override
  _FoodMenumenuScreenState createState() => _FoodMenumenuScreenState();
}

class _FoodMenumenuScreenState extends State<FoodMenumenuScreen> {
  // Stream for fetching food menu data
  Stream<QuerySnapshot> getFoodMenu() {
    return FirebaseFirestore.instance.collection('Foods').snapshots();
  }

  void _toggleMenuAvailability(String docId, bool newStatus) async {
    try {
      await FirebaseFirestore.instance.collection('Foods').doc(docId).update({
        'isAvailable': newStatus, // ✅ อัปเดตสถานะ
      });

      showMenuStatusDialog(context, newStatus);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("เกิดข้อผิดพลาด: $e")),
      );
    }
  }

  void showManageCategoriesDialog(BuildContext context) {
    TextEditingController categoryController = TextEditingController();

    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return Dialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20.0),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(15),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // ✅ Header ของป๊อปอัป
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.blue,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.category, color: Colors.white),
                      SizedBox(width: 8),
                      Text(
                        "จัดการประเภทอาหาร",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(height: 15),

                // ✅ แสดงประเภทอาหารจาก Firestore
                StreamBuilder<QuerySnapshot>(
                  stream: FirebaseFirestore.instance
                      .collection('Categories')
                      .snapshots(),
                  builder: (context, snapshot) {
                    if (!snapshot.hasData) return CircularProgressIndicator();
                    var categories = snapshot.data!.docs;

                    if (categories.isEmpty) {
                      return Text("ไม่มีประเภทอาหาร",
                          style: TextStyle(color: Colors.grey));
                    }

                    return Container(
                      height: 250, // ✅ กำหนดความสูงให้เหมาะสม
                      child: ListView.separated(
                        shrinkWrap: true,
                        itemCount: categories.length,
                        separatorBuilder: (context, index) =>
                            Divider(color: Colors.grey[300]),
                        itemBuilder: (context, index) {
                          var category = categories[index];

                          return Card(
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10)),
                            elevation: 2,
                            child: ListTile(
                              contentPadding: EdgeInsets.symmetric(
                                  vertical: 10, horizontal: 15),
                              title: Text(category['name'],
                                  style: TextStyle(fontSize: 16)),
                              trailing: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  // ✏ ปุ่มแก้ไข
                                  IconButton(
                                    icon: Icon(Icons.edit, color: Colors.blue),
                                    onPressed: () {
                                      categoryController.text =
                                          category['name'];
                                      showEditCategoryDialog(context,
                                          category.id, categoryController);
                                    },
                                  ),

                                  // 🗑 ปุ่มลบ
                                  IconButton(
                                    icon: Icon(Icons.delete, color: Colors.red),
                                    onPressed: () =>
                                        deleteCategory(category.id),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                    );
                  },
                ),
                SizedBox(height: 15),

                // ✅ ปุ่มเพิ่มประเภท
                ElevatedButton.icon(
                  onPressed: () {
                    showAddCategoryDialog(context);
                  },
                  icon: Icon(Icons.add, color: Colors.white),
                  label: Text("เพิ่มประเภท"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10)),
                    padding: EdgeInsets.symmetric(horizontal: 30, vertical: 12),
                  ),
                ),

                SizedBox(height: 10),

                // ✅ ปุ่มปิด
                TextButton(
                  onPressed: () => Navigator.pop(dialogContext),
                  child: Text("❌ ปิด",
                      style: TextStyle(color: Colors.red, fontSize: 16)),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  void showAddCategoryDialog(BuildContext context) {
    TextEditingController categoryController = TextEditingController();

    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: Text("➕ เพิ่มประเภทอาหาร"),
          content: TextField(
            controller: categoryController,
            decoration: InputDecoration(hintText: "ชื่อประเภท"),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: Text("ยกเลิก"),
            ),
            ElevatedButton(
              onPressed: () async {
                if (categoryController.text.isNotEmpty) {
                  await FirebaseFirestore.instance
                      .collection('Categories')
                      .add({
                    'name': categoryController.text,
                    'createdAt': FieldValue.serverTimestamp(),
                  });

                  Navigator.pop(dialogContext);
                }
              },
              child: Text("เพิ่ม"),
            ),
          ],
        );
      },
    );
  }

  void showEditCategoryDialog(BuildContext context, String categoryId,
      TextEditingController controller) {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: Text("✏ แก้ไขประเภทอาหาร"),
          content: TextField(
            controller: controller,
            decoration: InputDecoration(hintText: "ชื่อประเภทใหม่"),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: Text("ยกเลิก"),
            ),
            ElevatedButton(
              onPressed: () async {
                if (controller.text.isNotEmpty) {
                  await FirebaseFirestore.instance
                      .collection('Categories')
                      .doc(categoryId)
                      .update({
                    'name': controller.text,
                    'updatedAt': FieldValue.serverTimestamp(),
                  });

                  Navigator.pop(dialogContext);
                }
              },
              child: Text("บันทึก"),
            ),
          ],
        );
      },
    );
  }

  void deleteCategory(String categoryId) async {
    await FirebaseFirestore.instance
        .collection('Categories')
        .doc(categoryId)
        .delete();
  }

  void showMenuStatusDialog(BuildContext context, bool newStatus) {
    showDialog(
      context: context,
      barrierDismissible: false, // Prevents closing by tapping outside
      builder: (BuildContext dialogContext) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20), // Rounded corners
          ),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  newStatus ? Icons.check_circle : Icons.cancel,
                  size: 60,
                  color: newStatus ? Colors.green : Colors.red,
                ),
                const SizedBox(height: 10),
                Text(
                  newStatus ? "เมนูเปิดใช้งานสำเร็จ!" : "เมนูถูกปิดสำเร็จ!",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: newStatus ? Colors.green[800] : Colors.red[800],
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 10),
                Text(
                  newStatus
                      ? "ลูกค้าสามารถสั่งซื้อเมนูนี้ได้แล้ว 🎉"
                      : "เมนูนี้ถูกปิดชั่วคราวและจะไม่แสดงให้ลูกค้าเห็น",
                  style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.pop(dialogContext); // Close popup
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: newStatus ? Colors.green : Colors.red,
                    padding: EdgeInsets.symmetric(horizontal: 30, vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: const Text(
                    "ตกลง",
                    style: TextStyle(fontSize: 16, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  void _deleteMenuItem(String docId) async {
    try {
      await FirebaseFirestore.instance.collection('Foods').doc(docId).delete();
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text("สำเร็จ"),
            content: const Text("ลบเมนูสำเร็จ"),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text("ตกลง"),
              ),
            ],
          );
        },
      );
    } catch (e) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text("เกิดข้อผิดพลาด"),
            content: Text("ไม่สามารถลบเมนูได้: $e"),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text("ปิด"),
              ),
            ],
          );
        },
      );
    }
  }

  /// **🔹 ฟังก์ชันแสดงภาพ (รองรับทั้ง Base64 & URL)**
  Widget _buildMenuImage(String imageData) {
    Uint8List? imageBytes;

    try {
      if (imageData.isNotEmpty) {
        // ถ้าเป็น Base64 (ไม่ใช่ URL)
        if (!imageData.startsWith('http')) {
          imageBytes = base64Decode(imageData);
        }
      }
    } catch (e) {
      print("Error decoding image: $e");
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(8.0),
      child: imageBytes != null
          ? Image.memory(imageBytes, width: 60, height: 60, fit: BoxFit.cover)
          : Image.network(
              imageData,
              width: 60,
              height: 60,
              fit: BoxFit.cover,
              errorBuilder: (context, error, stackTrace) =>
                  const Icon(Icons.broken_image, size: 60, color: Colors.grey),
            ),
    );
  }

  @override
  Widget _buildToggleSwitch(String docId, bool isAvailable) {
    return StreamBuilder<DocumentSnapshot>(
      stream:
          FirebaseFirestore.instance.collection('Foods').doc(docId).snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator();
        }

        if (snapshot.hasError || !snapshot.hasData || !snapshot.data!.exists) {
          return const Icon(Icons.error, color: Colors.red);
        }

        bool currentStatus = snapshot.data!.get('isAvailable') ??
            true; // ✅ อ่านค่าล่าสุดจาก Firestore

        return Switch(
          value: currentStatus, // ✅ ใช้ค่าจาก Firestore
          onChanged: (bool newValue) {
            _toggleMenuAvailability(docId, newValue); // ✅ อัปเดต Firestore
          },
          activeColor: Colors.green, // ✅ สีเขียวถ้าเปิด
          inactiveTrackColor: Colors.grey, // ✅ สีเทาถ้าปิด
        );
      },
    );
  }

  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60),
        child: ClipRRect(
          borderRadius: const BorderRadius.only(
            bottomLeft: Radius.circular(20),
            bottomRight: Radius.circular(20),
          ),
          child: AppBar(
            flexibleSpace: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(0.5),
                    Colors.orangeAccent,
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
            title: const Text(
              "เมนูอาหาร",
              style: TextStyle(
                fontFamily: "assets/fonts/ChakraPetch-Bold.ttf",
                color: Color.fromARGB(255, 0, 0, 0),
                fontWeight: FontWeight.bold,
              ),
            ),
            centerTitle: true,
            elevation: 5,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                // ✅ ปุ่มเพิ่มเมนู
                ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => AddItemsScreen()),
                    );
                  },
                  icon: Icon(Icons.add, color: Colors.white),
                  label: Text("จัดการเมนู"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10)),
                  ),
                ),

                SizedBox(width: 10), // ✅ เว้นระยะห่างระหว่างปุ่ม

                // ✅ ปุ่มจัดการประเภท
                ElevatedButton.icon(
                  onPressed: () {
                    showManageCategoriesDialog(
                        context); // ✅ ใช้ป๊อปอัปแทนการเปิดหน้าใหม่
                  },
                  icon: Icon(Icons.category, color: Colors.white),
                  label: Text("จัดการประเภท"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: StreamBuilder<QuerySnapshot>(
                stream: getFoodMenu(),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  if (snapshot.hasError) {
                    return Center(child: Text("Error: ${snapshot.error}"));
                  }
                  if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                    return const Center(child: Text("ไม่มีเมนู"));
                  }
                  final menuItems = snapshot.data!.docs.map((doc) {
                    final data = doc.data() as Map<String, dynamic>;
                    return {
                      "name": data['name'] ?? "ไม่มีชื่อเมนู",
                      "imageBase64": data['imageBase64'] ?? "",
                      "price": data['price']?.toString() ?? "0",
                      "category": data['category'] ?? "ไม่ระบุหมวดหมู่",
                      "isAvailable": data['isAvailable'] ??
                          true, // ✅ กำหนดค่า Default เป็น true
                    };
                  }).toList();

                  // Parse food menu data
                  Map<String, List<Map<String, dynamic>>> foodMenu = {};

                  for (var doc in snapshot.data!.docs) {
                    final data = doc.data() as Map<String, dynamic>;
                    String docId = doc.id;
                    String category = data['category'] ?? 'ไม่ระบุหมวดหมู่';
                    String name = data['name'] ?? 'ไม่มีชื่อเมนู';
                    String price = data['price'] ?? '0';
                    String image = data['imageBase64'] ?? ''; // ✅ ดึงรูป Base64

                    if (!foodMenu.containsKey(category)) {
                      foodMenu[category] = [];
                    }

                    foodMenu[category]!.add({
                      'docId': docId,
                      'name': name,
                      'price': price,
                      'image': image,
                    });
                  }

                  return ListView.builder(
                    itemCount: foodMenu.keys.length,
                    itemBuilder: (context, index) {
                      String category = foodMenu.keys.elementAt(index);
                      List<Map<String, dynamic>> items = foodMenu[category]!;

                      return Card(
                        elevation: 4,
                        margin: const EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12.0),
                        ),
                        child: ExpansionTile(
                          title: Text(
                            "$category (${items.length})",
                            style: const TextStyle(
                                fontFamily: "IBMPlexSansThai",
                                fontWeight: FontWeight.bold),
                          ),
                          children: items.map((item) {
                            bool isAvailable = item["isAvailable"] ??
                                true; // ค่าเริ่มต้นเป็นเปิด

                            return Padding(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 8.0, horizontal: 16.0),
                              child: Row(
                                children: [
                                  _buildMenuImage(item["image"]),
                                  const SizedBox(width: 16),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          item["name"],
                                          style: TextStyle(
                                            fontFamily: "IBMPlexSansThai",
                                            fontSize: 16,
                                            fontWeight: FontWeight.w500,
                                            color: isAvailable
                                                ? Colors.black
                                                : Colors.grey, // สีจางลงถ้าหมด
                                          ),
                                        ),
                                        Text(
                                          "${item["price"]} บาท",
                                          style: TextStyle(
                                            fontFamily: "IBMPlexSansThai",
                                            fontSize: 14,
                                            color: isAvailable
                                                ? Colors.green
                                                : Colors
                                                    .red, // เปลี่ยนสีราคาถ้าหมด
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  _buildToggleSwitch(item['docId'],
                                      isAvailable), // ✅ เพิ่ม Toggle Switch
                                  IconButton(
                                    icon: const Icon(Icons.edit,
                                        color: Colors.blue),
                                    onPressed: () {
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (context) => EditItemScreen(
                                            menuId: item['docId'],
                                            menuData: {
                                              'name': item['name'],
                                              'price': item['price'],
                                              'image': item['image'],
                                              'isAvailable':
                                                  item['isAvailable'],
                                            },
                                          ),
                                        ),
                                      );
                                    },
                                  ),
                                  IconButton(
                                    icon: const Icon(Icons.delete,
                                        color: Colors.red),
                                    onPressed: () {
                                      _deleteMenuItem(item['docId']);
                                    },
                                  ),
                                ],
                              ),
                            );
                          }).toList(),
                        ),
                      );
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
