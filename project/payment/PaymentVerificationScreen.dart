import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class PaymentVerificationScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("ตรวจสอบการชำระเงิน", style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.orangeAccent,
        centerTitle: true,
        elevation: 5,
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance
            .collection('Orders')
            .where('paymentStatus', isEqualTo: 'รอการตรวจสอบ') // ✅ ดึงเฉพาะรายการที่ต้องตรวจสอบ
            .snapshots(),
        builder: (context, snapshot) {
          if (!snapshot.hasData) return Center(child: CircularProgressIndicator());

          final orders = snapshot.data!.docs;
          if (orders.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.check_circle_outline, color: Colors.green, size: 80),
                  SizedBox(height: 10),
                  Text(
                    "ไม่มีรายการรอตรวจสอบ",
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.grey),
                  ),
                ],
              ),
            );
          }

          return ListView.builder(
            padding: EdgeInsets.all(16),
            itemCount: orders.length,
            itemBuilder: (context, index) {
              final orderData = orders[index].data() as Map<String, dynamic>;
              final orderId = orders[index].id;
              final totalPrice = orderData['totalPrice'] ?? 0;
              final userId = orderData['userId']; // ✅ ดึง userId ของลูกค้า
              final Uint8List? imageBytes = orderData['paymentSlip'] != null
                  ? base64Decode(orderData['paymentSlip'])
                  : null;

              return FutureBuilder<DocumentSnapshot>(
                future: FirebaseFirestore.instance.collection('Customers').doc(userId).get(),
                builder: (context, customerSnapshot) {
                  if (!customerSnapshot.hasData) {
                    return Center(child: CircularProgressIndicator());
                  }

                  final customerData = customerSnapshot.data!.data() as Map<String, dynamic>?;

                  final customerName = customerData?['name'] ?? "ไม่พบข้อมูล";
                  final customerSurname = customerData?['surname'] ?? "";
                  final customerAddress = customerData?['address'] ?? "ไม่พบที่อยู่";
                  final customerPhone = customerData?['phone'] ?? "ไม่พบเบอร์โทร";

                  return Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                    margin: EdgeInsets.only(bottom: 16),
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                "Order ID: $orderId",
                                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                              ),
                              Chip(
                                label: Text("รอการตรวจสอบ"),
                                backgroundColor: Colors.amberAccent,
                                labelStyle: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                          SizedBox(height: 10),
                          Text("ยอดรวม: ฿$totalPrice", style: TextStyle(fontSize: 16, color: Colors.green)),
                          SizedBox(height: 15),

                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // 🔹 ภาพสลิปทางซ้าย
                              Expanded(
                                flex: 1,
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(10),
                                  child: imageBytes != null
                                      ? Image.memory(imageBytes, height: 200, fit: BoxFit.cover)
                                      : Container(
                                          height: 200,
                                          color: Colors.grey[300],
                                          child: Center(child: Text("ไม่มีสลิปแนบมา")),
                                        ),
                                ),
                              ),

                              SizedBox(width: 16),

                              // 🔹 ข้อมูลลูกค้าทางขวา
                              Expanded(
                                flex: 1,
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        Icon(Icons.person, color: Colors.blue),
                                        SizedBox(width: 8),
                                        Expanded(
                                          child: Text(
                                            "ชื่อ: $customerName $customerSurname",
                                            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                                          ),
                                        ),
                                      ],
                                    ),
                                    SizedBox(height: 8),
                                    Row(
                                      children: [
                                        Icon(Icons.home, color: Colors.green),
                                        SizedBox(width: 8),
                                        Expanded(
                                          child: Text(
                                            "ที่อยู่: $customerAddress",
                                            style: TextStyle(fontSize: 14),
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                        ),
                                      ],
                                    ),
                                    SizedBox(height: 8),
                                    Row(
                                      children: [
                                        Icon(Icons.phone, color: Colors.redAccent),
                                        SizedBox(width: 8),
                                        Text(
                                          "โทร: $customerPhone",
                                          style: TextStyle(fontSize: 14),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),

                          SizedBox(height: 15),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              ElevatedButton.icon(
                                onPressed: () => _confirmPayment(context, orderId),
                                icon: Icon(Icons.check_circle, color: Colors.white),
                                label: Text("อนุมัติ"),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.green,
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                                ),
                              ),
                              ElevatedButton.icon(
  onPressed: () => _rejectPayment(context, orderId),
  icon: Icon(Icons.cancel, color: Colors.white),
  label: Text("ปฏิเสธ"),
  style: ElevatedButton.styleFrom(
    backgroundColor: Colors.red,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
  ),
)

                            ],
                          ),
                        ],
                      ),
                    ),
                  );
                },
              );
            },
          );
        },
      ),
    );
  }

  void _confirmPayment(BuildContext context, String orderId) async {
    await FirebaseFirestore.instance.collection('Orders').doc(orderId).update({
      'paymentStatus': 'ตรวจสอบแล้ว',
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("✅ อนุมัติการชำระเงินเรียบร้อย")),
    );
  }

 void _rejectPayment(BuildContext context, String orderId) async {
  try {
    // อัปเดตสถานะใน Firestore ให้เป็น "Cancelled"
    await FirebaseFirestore.instance.collection('Orders').doc(orderId).update({
      'status': 'Cancelled', // เปลี่ยนสถานะเป็น "ยกเลิก"
      'paymentStatus': 'Rejected', // อัปเดตสถานะการชำระเงิน (ถ้ามี)
    });

    // แจ้งเตือนการอัปเดตสถานะ
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("ออเดอร์ $orderId ถูกยกเลิกเรียบร้อยแล้ว")),
    );

    // ถ้าต้องการปิด dialog หรือทำการนำทางกลับ
    Navigator.pop(context); // ปิด dialog หรือหน้าเดิม
  } catch (e) {
    // จัดการข้อผิดพลาด
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("เกิดข้อผิดพลาด: $e")),
    );
  }
}

}
