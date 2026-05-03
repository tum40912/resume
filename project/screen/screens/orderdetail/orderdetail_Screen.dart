import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:qr_flutter/qr_flutter.dart';

class OrderDetailScreen extends StatelessWidget {
  final Map<String, dynamic> order;
  final String customerName;
  final String customerPhone;
  final String customerAddress;
  final String orderId; // รับ Document ID

  const OrderDetailScreen({
    Key? key,
    required this.order,
    required this.customerName,
    required this.customerPhone,
    required this.customerAddress,
    required this.orderId, // เพิ่มพารามิเตอร์นี้
  }) : super(key: key);

// ✅ อัปเดตสถานะและปิดป๊อปอัป
   Future<void> updateOrderStatus(BuildContext context) async {
    try {
      await FirebaseFirestore.instance
          .collection('Orders')
          .doc(orderId)
           .update({'status': 'Delivered'});

      Navigator.pop(context);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("เกิดข้อผิดพลาด: $e")),
      );
    }
  }

// ✅ แสดงตัวเลือกชำระเงิน
  

// ✅ แสดง QR Code จากไฟล์ใน assets และเพิ่มปุ่ม "เสร็จสิ้น"
  

  // Future<void> updateOrderStatus(BuildContext context) async {
  //   try {
  //     // ใช้ orderId เพื่ออัปเดตสถานะใน Firestore
  //     await FirebaseFirestore.instance
  //         .collection('Orders')
  //         .doc(orderId) // ใช้ Document ID จาก Firestore
  //         .update({'status': 'Delivered'});

  //     Navigator.pop(context);
  //   } catch (e) {
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       SnackBar(content: Text("เกิดข้อผิดพลาด: $e")),
  //     );
  //   }
  // }

  void showDeliverySuccessDialog(BuildContext context) {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยการกดนอกกรอบ
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20), // ขอบโค้งมน
          ),
          title: Row(
            children: const [
              Icon(Icons.check_circle, color: Colors.green, size: 28),
              SizedBox(width: 8),
              Text(
                "จัดส่งสำเร็จ!",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          content: const Text(
            "รายการอาหารได้ถูกจัดส่งถึงลูกค้าเรียบร้อยแล้ว\nขอบคุณที่ให้บริการ!",
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16),
          ),
          actionsAlignment: MainAxisAlignment.center,
          actions: [
            ElevatedButton(
              onPressed: () {
                Navigator.pop(dialogContext); // ปิดป็อปอัป
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: const Text(
                "ตกลง",
                style:
                    TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        );
      },
    );
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
              "รายละเอียดการจัดส่ง",
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
        width: double.infinity, // กำหนดความกว้างให้เต็มจอ
        height: double.infinity, // กำหนดความสูงให้เต็มจอ
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color.fromARGB(255, 240, 239, 237)
                  .withOpacity(0.5), // สีส้มไล่เฉด
              const Color.fromARGB(255, 252, 213, 162),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ข้อมูลลูกค้า
              SizedBox(
                width: double.infinity, // ✅ ทำให้ Card กว้างเต็มจอ
                child: Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.person,
                                color: Colors.blue, size: 24), // ✅ ไอคอนลูกค้า
                            SizedBox(width: 8),
                            Text(
                              customerName,
                              style: const TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            Icon(Icons.phone,
                                color: Colors.green,
                                size: 24), // ✅ ไอคอนเบอร์โทร
                            SizedBox(width: 8),
                            Text(
                              customerPhone,
                              style: const TextStyle(fontSize: 16),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(Icons.location_on,
                                color: Colors.red, size: 24), // ✅ ไอคอนที่อยู่
                            SizedBox(width: 8),
                            Expanded(
                              // ✅ ป้องกันข้อความที่อยู่ล้น
                              child: Text(
                                customerAddress,
                                style: const TextStyle(fontSize: 16),
                                softWrap: true, // ✅ รองรับข้อความยาว
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // รายการอาหารทั้งหมด
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "รายการอาหาร",
                        style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.orange),
                      ),
                      const SizedBox(height: 10),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: (order['items'] as List).map((item) {
                          final List<dynamic>? sides =
                              item['sides'] as List<dynamic>?;
                          final String note =
                              item['note'] ?? ''; // ✅ ดึงค่า note

                          return Padding(
                            padding: const EdgeInsets.only(bottom: 8.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Expanded(
                                      child: Text(
                                        "${item['name']} (x${item['quantity']})",
                                        style: const TextStyle(fontSize: 16),
                                      ),
                                    ),
                                    Text(
                                      "${int.parse(item['price']) * item['quantity']} บาท", // ✅ แปลง String เป็น int ก่อนคำนวณ
                                      style: const TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.green,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 5),
                                if (sides != null && sides.isNotEmpty) ...[
                                  const Text(
                                    "เพิ่มเติม:",
                                    style: TextStyle(
                                        fontWeight: FontWeight.bold,
                                        color: Colors.grey),
                                  ),
                                  ...sides.map((side) {
                                    return Padding(
                                      padding:
                                          const EdgeInsets.only(left: 16.0),
                                      child: Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        children: [
                                          Expanded(
                                            child: Text(
                                              "- ${side['name']}",
                                              style:
                                                  const TextStyle(fontSize: 14),
                                            ),
                                          ),
                                          Text(
                                            "+${side['price']} บาท",
                                            style: const TextStyle(
                                                fontSize: 14,
                                                fontWeight: FontWeight.bold,
                                                color: Colors.grey),
                                          ),
                                        ],
                                      ),
                                    );
                                  }).toList(),
                                ],
                                if (note.isNotEmpty) ...[
                                  // ✅ แสดงหมายเหตุถ้ามี
                                  const Padding(
                                    padding:
                                        EdgeInsets.only(left: 16.0, top: 4.0),
                                    child: Text(
                                      'หมายเหตุ:',
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold,
                                          color: Colors.red),
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.only(left: 32.0),
                                    child: Text(
                                      note,
                                      style: const TextStyle(
                                          fontSize: 14,
                                          fontStyle: FontStyle.italic,
                                          color: Color.fromARGB(255, 0, 0, 0)),
                                    ),
                                  ),
                                  SizedBox(
                                      height:
                                          8.0), // ✅ เพิ่มระยะห่างระหว่างข้อความด้านบน
                                  Text(
                                    "ยอดรวม: ${order['totalPrice']} บาท",
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.green,
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          );
                        }).toList(),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 20),

              // ✅ ปุ่ม "จัดส่งสำเร็จ" -> แสดงตัวเลือกชำระเงิน
              Align(
                alignment: Alignment.center,
                child: ElevatedButton(
                  onPressed: () async {
                    await updateOrderStatus(context); // อัปเดตสถานะ
                    showDeliverySuccessDialog(context); // แสดงป็อปอัป
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                     padding: const EdgeInsets.symmetric(
                        vertical: 12, horizontal: 50),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                 child: const Text(
                    "จัดส่งสำเร็จ",
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
