import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class OrderScreen extends StatelessWidget {
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
              "ออเดอร์",
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
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance.collection('Orders').where('status',
                whereIn: [
              'Waiting',
              'Accepted'
            ]) // แสดงเฉพาะสถานะ Waiting และ Accepted
            .snapshots(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return const Center(
              child: Text('ไม่มีออเดอร์'),
            );
          }
          final orders = snapshot.data!.docs;
          return ListView.builder(
            padding: const EdgeInsets.all(8.0),
            itemCount: orders.length,
            itemBuilder: (context, index) {
              final orderData = orders[index].data() as Map<String, dynamic>;
              final customerId = orderData['userId'] as String?;
              final createdAt = orderData['timestamp'] != null
                  ? (orderData['timestamp'] as Timestamp).toDate()
                  : null;
              final status = orderData['status'] ?? 'Unknown';
              final orderItems = (orderData['items'] as List<dynamic>? ?? [])
                  .map((item) => {
                        'name': item['name'] ?? 'ไม่มีชื่อสินค้า',
                        'price': item['price'],
                        'quantity': item['quantity'] ?? 1,
                        'sides': item['sides'] ?? [],
                        'note': item['note'] ?? '', // ✅ ดึงค่า note มาด้วย
                      })
                  .toList();

              return FutureBuilder<DocumentSnapshot>(
                future: customerId != null
                    ? FirebaseFirestore.instance
                        .collection('Customers')
                        .doc(customerId)
                        .get()
                    : Future.value(null),
                builder: (context, customerSnapshot) {
                  if (customerSnapshot.connectionState ==
                      ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  }

                  if (!customerSnapshot.hasData ||
                      !customerSnapshot.data!.exists) {
                    return OrderCard(
                      customerName: 'ไม่มีข้อมูล (userId: $customerId)',
                      resortName: 'ไม่มีข้อมูล',
                      createdAt: createdAt,
                      phone: 'ไม่มีข้อมูล',
                      status: status,
                      orders: orderItems,
                      orderId: orders[index].id,
                    );
                  }

                  final customerData =
                      customerSnapshot.data!.data() as Map<String, dynamic>;
                  final customerName = customerId != null
                      ? (customerData['name'] ?? 'ไม่มีข้อมูล')
                      : 'ไม่พบข้อมูลลูกค้า';

                  final resortName = customerData['address'] ?? 'ไม่มีข้อมูล';
                  final phone = customerData['phone'] ?? 'ไม่มีข้อมูล';

                  return OrderCard(
                    customerName: customerName,
                    resortName: resortName,
                    createdAt: createdAt,
                    phone: phone,
                    status: status,
                    orders: orderItems,
                    orderId: orders[index].id,
                  );
                },
              );
            },
          );
        },
      ),
    );
  }
}

Future<void> _confirmUpdateStatus(
    BuildContext context, String orderId, String currentStatus) async {
  String nextStatus = currentStatus == 'Waiting' ? 'Accepted' : 'Completed';
  String actionText = currentStatus == 'Waiting' ? 'รับออเดอร์' : 'เสร็จสิ้น';

  showDialog(
    context: context,
    barrierDismissible: false, // ต้องกดปุ่มเท่านั้นถึงจะปิด
    builder: (BuildContext dialogContext) {
      return AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
        title: Row(
          children: [
            Icon(Icons.info_outline, color: Colors.blue, size: 30),
            SizedBox(width: 10),
            Text("ยืนยันการเปลี่ยนสถานะ"),
          ],
        ),
        content:
            Text("คุณแน่ใจหรือไม่ว่าต้องการเปลี่ยนออเดอร์เป็น '$nextStatus'?"),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(dialogContext); // ปิดป๊อปอัป
            },
            child: Text("ยกเลิก", style: TextStyle(color: Colors.grey)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.blue),
            onPressed: () async {
              Navigator.pop(dialogContext); // ปิดป๊อปอัป
              await FirebaseFirestore.instance
                  .collection('Orders')
                  .doc(orderId)
                  .update({'status': nextStatus});

              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                    content:
                        Text("อัปเดตสถานะออเดอร์เป็น '$nextStatus' เรียบร้อย")),
              );
            },
            child: Text(actionText, style: TextStyle(color: Colors.white)),
          ),
        ],
      );
    },
  );
}

Future<void> _confirmCancelOrder(BuildContext context, String orderId) async {
  showDialog(
    context: context,
    barrierDismissible: false,
    builder: (BuildContext dialogContext) {
      return AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.red, size: 30),
            SizedBox(width: 10),
            Text("ยกเลิกออเดอร์"),
          ],
        ),
        content: Text("คุณแน่ใจหรือไม่ว่าต้องการยกเลิกออเดอร์นี้?"),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(dialogContext);
            },
            child: Text("ยกเลิก", style: TextStyle(color: Colors.grey)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () async {
              Navigator.pop(dialogContext);
              await FirebaseFirestore.instance
                  .collection('Orders')
                  .doc(orderId)
                  .update({'status': 'Cancelled'});

              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("ออเดอร์ถูกยกเลิกเรียบร้อย")),
              );
            },
            child: Text("ยืนยัน", style: TextStyle(color: Colors.white)),
          ),
        ],
      );
    },
  );
}

class OrderCard extends StatelessWidget {
  final String customerName;
  final String resortName;
  final DateTime? createdAt;
  final String phone;
  final String status;
  final List<Map<String, dynamic>> orders;
  final String orderId;

  OrderCard({
    required this.customerName,
    required this.resortName,
    required this.createdAt,
    required this.phone,
    required this.status,
    required this.orders,
    required this.orderId,
  });

  // ย้าย parsePrice มาที่นี่
  double parsePrice(dynamic price) {
    if (price is int) {
      return price.toDouble();
    } else if (price is String) {
      return double.tryParse(price) ?? 0.0;
    } else {
      return 0.0;
    }
  }

  Color _getStatusColor(String status) {
    switch (status) {
      case 'Waiting':
        return Colors.orange;
      case 'Accepted':
        return Colors.blue;
      case 'Completed':
        return Colors.green;
      case 'Cancelled':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  String translateStatus(String status) {
    switch (status) {
      case 'Waiting':
        return 'รอรับออเดอร์';
      case 'Accepted':
        return 'กำลังปรุง';
      case 'Completed':
        return 'ปรุงเสร็จสิ้น';
      case 'Cancelled':
        return 'ยกเลิก';
      default:
        return 'ไม่ทราบสถานะ';
    }
  }

  @override
  Widget build(BuildContext context) {
    final double totalPrice = orders.fold(0.0, (sum, item) {
      final itemPrice = parsePrice(item['price']);
      final sidesPrice = (item['sides'] as List<dynamic>)
          .fold(0.0, (sSum, side) => sSum + parsePrice(side['price']));
      return sum + (itemPrice + sidesPrice) * (item['quantity'] ?? 1);
    });

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: ExpansionTile(
        leading: const Icon(Icons.person, color: Colors.orange),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'ลูกค้า: $customerName',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            Text(
              'สถานะ: ${translateStatus(status)}',
              style: TextStyle(
                color: _getStatusColor(status),
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Order ID: $orderId',
              style: const TextStyle(
                fontSize: 14,
                color: Colors.grey,
              ),
            ),
          ],
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('รีสอร์ท: $resortName'),
            if (createdAt != null)
              Text('เวลา: ${createdAt.toString().substring(0, 16)}'),
            Text('เบอร์โทร: $phone'),
          ],
        ),
        children: [
          ...orders.map((order) {
            final sides = order['sides'] as List<dynamic>;
            final note = order['note'];
            final quantity = order['quantity'] ??
                1; // ดึงค่าจำนวน (quantity) ถ้าไม่มีใช้ค่าเริ่มต้น 1

            return Padding(
              padding:
                  const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
              child: Card(
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // ชื่อเมนู
                      ListTile(
                        title: Text(
                          "${order['name']} x$quantity", // แสดงชื่อสินค้าและจำนวน
                          style: const TextStyle(
                              fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                        trailing: Text(
                          '${(int.parse(order['price'].toString()) * quantity).toString()} บาท', // แปลงราคาจาก String เป็น int
                          style: const TextStyle(
                              color: Colors.green, fontWeight: FontWeight.bold),
                        ),
                      ),

                      // เพิ่มเติม
                      if (sides.isNotEmpty) ...[
                        const Text(
                          "📌 เพิ่มเติม:",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.pink,
                          ),
                        ),
                        ...sides.map((side) {
                          final sideName = side['name'] ?? 'ไม่มีชื่อ';
                          final sidePrice = side['price'] ?? 0;
                          return Padding(
                            padding: const EdgeInsets.only(left: 16.0),
                            child: Text(
                                "- $sideName (+${sidePrice.toString()} บาท)"),
                          );
                        }).toList(),
                        const SizedBox(height: 8),
                      ],

                      // หมายเหตุ
                      if (note.isNotEmpty) ...[
                        const Text(
                          "📝 หมายเหตุ:",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.red,
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(left: 16.0),
                          child: Text(note),
                        ),
                        const SizedBox(height: 8),
                      ],

                      // ราคาเมนู
                      // Align(
                      //   alignment: Alignment.centerRight,
                      //   child: Text(
                      //     "${order['price']} บาท",
                      //     style: const TextStyle(
                      //       fontSize: 16,
                      //       fontWeight: FontWeight.bold,
                      //       color: Colors.green,
                      //     ),
                      //   ),
                      // ),
                    ],
                  ),
                ),
              ),
            );
          }).toList(),

          // เพิ่มส่วนราคารวม
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'ราคารวม: ${totalPrice.toStringAsFixed(0)} บาท',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.green,
                  ),
                ),
              ],
            ),
          ),

          // ปุ่มยกเลิกและอัปเดตสถานะ
          Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                if (status == 'Waiting') // แสดงปุ่มยกเลิกเฉพาะ Waiting
                  TextButton(
                    onPressed: () => _confirmCancelOrder(context, orderId),
                    child: const Text("ยกเลิก",
                        style: TextStyle(color: Colors.red)),
                  ),
                ElevatedButton(
                  onPressed: () =>
                      _confirmUpdateStatus(context, orderId, status),
                  style: ElevatedButton.styleFrom(
                    backgroundColor:
                        status == 'Waiting' ? Colors.orange : Colors.green,
                  ),
                  child: Text(translateStatus(
                      status == 'Waiting' ? 'Accepted' : 'Completed')),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
