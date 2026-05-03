import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart';
import 'package:krua_pa_ree/screens/history/history_screen.dart';

class ReportScreen extends StatefulWidget {
  @override
  _ReportScreenState createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  String? selectedDay;
  String? selectedMonth;
  int currentPageItems = 1; // แยก currentPage สำหรับสินค้าที่ขายรวมทั้งหมด
  int currentPageReviews = 1; // แยก currentPage สำหรับรีวิวจากลูกค้า
  final int currentPage = 1; // หน้าที่แสดงอยู่
  final int itemsPerPage = 10; // จำนวนเมนูต่อหน้า
  final int reviewsPerPage = 5;

  // สรุปรายงาน
  Future<Map<String, dynamic>> fetchOrdersReport() async {
    double totalSales = 0;
    Map<String, int> itemCounts = {};
    Map<String, double> dailySales = {};
    Map<String, double> monthlySales = {};

    QuerySnapshot snapshot = await FirebaseFirestore.instance
        .collection('Orders')
        .where('status', isEqualTo: 'Payment Completed') // กรองสถานะ Succeed
        .get();

    for (var doc in snapshot.docs) {
      final order = doc.data() as Map<String, dynamic>;
      final timestamp = order['timestamp'] as Timestamp;
      final totalPrice = (order['totalPrice'] ?? 0).toDouble();
      final items = order['items'] ?? [];

      totalSales += totalPrice;

      final date =
          DateTime.fromMillisecondsSinceEpoch(timestamp.millisecondsSinceEpoch);
      final dayKey = "${date.year}-${date.month}-${date.day}";
      final monthKey = "${date.year}-${date.month}";

      dailySales[dayKey] = (dailySales[dayKey] ?? 0) + totalPrice;
      monthlySales[monthKey] = (monthlySales[monthKey] ?? 0) + totalPrice;

      for (var item in items) {
        final name = item['name'];
        final quantity = (item['quantity'] ?? 0) as num;
        itemCounts[name] = (itemCounts[name] ?? 0) + quantity.toInt();
      }
    }

    return {
      'totalSales': totalSales,
      'itemCounts': itemCounts,
      'dailySales': dailySales,
      'monthlySales': monthlySales,
    };
  }

  // รีวิวหรือดาว
  Future<List<Map<String, dynamic>>> fetchReviews() async {
    QuerySnapshot snapshot = await FirebaseFirestore.instance
        .collection('Orders')
        .where('status', isEqualTo: 'Payment Completed') // กรองเฉพาะออเดอร์ที่สำเร็จ
        .get();

    List<Map<String, dynamic>> reviews = [];

    for (var doc in snapshot.docs) {
      final Map<String, dynamic> order =
          doc.data() as Map<String, dynamic>; // แปลงเป็น Map
      final String userId = order['userId'] ?? "unknown_user"; // ดึง userId

      // 🔹 ตรวจสอบว่ามีรีวิวและคะแนนใน Orders หรือไม่
      if (!order.containsKey('review') || !order.containsKey('star')) {
        continue; // ถ้าไม่มีรีวิวหรือดาว ข้ามไป
      }

      // 🔹 ดึงชื่อของลูกค้าจาก Collection "Customers"
      String customerName = "ไม่ทราบชื่อ"; // ค่าเริ่มต้น
      DocumentSnapshot customerDoc = await FirebaseFirestore.instance
          .collection('Customers')
          .doc(userId)
          .get();
      if (customerDoc.exists) {
        final Map<String, dynamic> customerData =
            customerDoc.data() as Map<String, dynamic>;
        customerName = customerData['name'] ?? "ไม่ทราบชื่อ";
      }

      // 🔹 เพิ่มข้อมูลรีวิวลงใน List
      reviews.add({
        'name': customerName, // ใช้ชื่อจริงของลูกค้า
        'review': order['review'], // รีวิวจาก Orders
        'star': order['star'], // คะแนนจาก Orders
      });
    }
    // print("📢 พบรีวิวทั้งหมด: ${reviews.length} รายการ");
    // for (var review in reviews) {
    //   print(
    //       "✅ รีวิว: ${review['review']} | ⭐ ${review['star']} | ลูกค้า: ${review['name']}");
    // }

    return reviews;
  }

  // report รายเดือน
  Future<List<Map<String, dynamic>>> fetchFoodMenuByMonth(String month) async {
    QuerySnapshot snapshot = await FirebaseFirestore.instance
        .collection('Orders')
        .where('status', isEqualTo: 'Payment Completed') // กรองสถานะ Succeed
        .get();

    List<Map<String, dynamic>> foodMenu = [];
    for (var doc in snapshot.docs) {
      final order = doc.data() as Map<String, dynamic>;
      final timestamp = order['timestamp'] as Timestamp;
      final orderDate =
          DateTime.fromMillisecondsSinceEpoch(timestamp.millisecondsSinceEpoch);

      final orderMonthString = "${orderDate.year}-${orderDate.month}";

      if (orderMonthString == month) {
        final items = order['items'] ?? [];
        for (var item in items) {
          final existingIndex = foodMenu
              .indexWhere((menuItem) => menuItem['name'] == item['name']);

          if (existingIndex == -1) {
            foodMenu.add({
              'name': item['name'],
              'quantity': (item['quantity'] ?? 0).toInt(),
            });
          } else {
            foodMenu[existingIndex]['quantity'] +=
                (item['quantity'] ?? 0).toInt();
          }
        }
      }
    }

    return foodMenu;
  }

  // ยอดขายประจำวัน
  Future<List<Map<String, dynamic>>> fetchFoodMenu(String date) async {
    QuerySnapshot snapshot = await FirebaseFirestore.instance
        .collection('Orders')
        .where('status', isEqualTo: 'Payment Completed') // กรองสถานะ Succeed
        .get();

    List<Map<String, dynamic>> foodMenu = [];
    for (var doc in snapshot.docs) {
      final order = doc.data() as Map<String, dynamic>;
      final timestamp = order['timestamp'] as Timestamp;
      final orderDate =
          DateTime.fromMillisecondsSinceEpoch(timestamp.millisecondsSinceEpoch);

      final orderDateString =
          "${orderDate.year}-${orderDate.month}-${orderDate.day}";

      if (orderDateString == date) {
        final items = order['items'] ?? [];
        for (var item in items) {
          final existingIndex = foodMenu
              .indexWhere((menuItem) => menuItem['name'] == item['name']);

          if (existingIndex == -1) {
            foodMenu.add({
              'name': item['name'],
              'quantity': (item['quantity'] ?? 0).toInt(),
            });
          } else {
            foodMenu[existingIndex]['quantity'] +=
                (item['quantity'] ?? 0).toInt();
          }
        }
      }
    }

    return foodMenu;
  }

  Map<String, Color> generateColors(List<String> menuNames) {
    final Map<String, Color> colorMap = {};
    final double step = 360 / menuNames.length; // ระยะห่างระหว่างเฉดสี

    for (int i = 0; i < menuNames.length; i++) {
      final hue = (i * step) % 360; // กำหนดเฉดสีที่ไม่ซ้ำ
      colorMap[menuNames[i]] = HSLColor.fromAHSL(1, hue, 0.7, 0.7).toColor();
    }

    return colorMap;
  }

  @override
  Widget buildLegend(Map<String, Color> colorMap) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: colorMap.entries.map((entry) {
        return Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: entry.value,
              ),
            ),
            SizedBox(width: 8),
            Text(
              entry.key,
              style: TextStyle(fontSize: 16),
            ),
          ],
        );
      }).toList(),
    );
  }

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
              "รายงาน",
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
      body: FutureBuilder<Map<String, dynamic>>(
        future: fetchOrdersReport(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('เกิดข้อผิดพลาด: ${snapshot.error}'));
          }

          final data = snapshot.data!;
          final totalSales = data['totalSales'] as double;
          final itemCounts = data['itemCounts'] as Map<String, int>;
          final dailySales = data['dailySales'] as Map<String, double>;
          final monthlySales = data['monthlySales'] as Map<String, double>;

          final dailyKeys = dailySales.keys.toList();
          final monthlyKeys = monthlySales.keys.toList();

          return Padding(
            padding: const EdgeInsets.all(16.0),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    width: double.infinity, // ให้ Card มีความกว้างเต็มหน้าจอ
                    child: StreamBuilder<QuerySnapshot>(
                      stream: FirebaseFirestore.instance
                          .collection('Orders')
                          .where('status',
                              isEqualTo: 'Payment Completed') // กรองสถานะ Succeed
                          .where('timestamp',
                              isGreaterThanOrEqualTo: Timestamp.fromDate(
                                DateTime(DateTime.now().year,
                                    DateTime.now().month, DateTime.now().day),
                              )) // เริ่มตั้งแต่เที่ยงคืนของวันนี้
                          .where('timestamp',
                              isLessThan: Timestamp.fromDate(
                                DateTime(
                                        DateTime.now().year,
                                        DateTime.now().month,
                                        DateTime.now().day)
                                    .add(Duration(days: 1)),
                              )) // ถึงเที่ยงคืนของวันถัดไป
                          .snapshots(),
                      builder: (context, snapshot) {
                        if (snapshot.connectionState ==
                            ConnectionState.waiting) {
                          return const Center(
                              child: CircularProgressIndicator());
                        }

                        if (snapshot.hasError) {
                          return Center(
                              child: Text('เกิดข้อผิดพลาด: ${snapshot.error}'));
                        }
                        if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                          return const Center(
                              child: Text('ไม่มีข้อมูลการขายสำหรับวันนี้'));
                        }

                        // คำนวณยอดขายรวม
                        final totalSales =
                            snapshot.data!.docs.fold<num>(0, (sum, doc) {
                          final data = doc.data()
                              as Map<String, dynamic>?; // ป้องกัน null
                          final totalPrice = data?['totalPrice'] as num? ??
                              0; // ตรวจสอบ totalPrice
                          return sum +
                              totalPrice; // เพิ่มค่า totalPrice ที่ไม่ใช่ null
                        });

                        return Card(
                          elevation: 4,
                          color: Colors.white,
                          child: Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'สรุปรายงาน',
                                  style: TextStyle(
                                    fontSize: 22,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.orange,
                                  ),
                                ),
                                SizedBox(height: 10),
                                Text(
                                  'ยอดขายวันนี้: ${totalSales.toInt()} บาท',
                                  style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold),
                                ),
                                SizedBox(height: 10),
                                Text(
                                  'วันที่: ${DateFormat('dd/MM/yyyy').format(DateTime.now())}',
                                  style: TextStyle(
                                      fontSize: 16, color: Colors.grey),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),

                  SizedBox(height: 16),
                  // ปุ่มประวัติออเดอร์
                  GestureDetector(
                    onTap: () {
                      // นำไปยังหน้าแสดงประวัติออเดอร์
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => HistoryScreen(),
                        ),
                      );
                    },
                    child: Card(
                      elevation: 4,
                      color: Colors.white,
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text(
                              "ดูประวัติออเดอร์",
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Icon(Icons.history, color: Colors.orange),
                          ],
                        ),
                      ),
                    ),
                  ),
                  Card(
                    elevation: 4,
                    color: Colors.white,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'อาหารที่ขายรวมทั้งหมด:',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.orange,
                            ),
                          ),
                          FutureBuilder(
                            future: Future.microtask(() {
                              return itemCounts.entries
                                  .map((e) => {
                                        'name': e.key,
                                        'count': e.value ?? 0
                                      }) // 🔥 Ensure count is never null
                                  .toList()
                                ..sort((a, b) => (b['count'] as int).compareTo(
                                    a['count'] as int)); // 🔥 Sort safely
                            }),
                            builder: (context,
                                AsyncSnapshot<List<Map<String, dynamic>>>
                                    snapshot) {
                              if (!snapshot.hasData)
                                return CircularProgressIndicator();

                              final sortedItems = snapshot.data!;

                              return Column(
                                children: [
                                  ListView.builder(
                                    shrinkWrap: true,
                                    physics: NeverScrollableScrollPhysics(),
                                    itemCount:
                                        ((currentPageItems - 1) * itemsPerPage +
                                                    itemsPerPage) >
                                                sortedItems.length
                                            ? sortedItems.length % itemsPerPage
                                            : itemsPerPage,
                                    itemBuilder: (context, index) {
                                      final actualIndex =
                                          (currentPageItems - 1) *
                                                  itemsPerPage +
                                              index;
                                      if (actualIndex >= sortedItems.length)
                                        return SizedBox();

                                      final itemName =
                                          sortedItems[actualIndex]['name'];
                                      final count = sortedItems[actualIndex]
                                          ['count'] as int;

                                      // ตรวจสอบว่าเป็นเครื่องดื่มหรือไม่
                                      bool isDrink = itemName.contains("ขวด") ||
                                          itemName.contains("กลม") ||
                                          itemName.contains("กาแฟ") ||
                                          itemName.contains("นม");

                                      return ListTile(
                                        title: Text(
                                          itemName,
                                          style: TextStyle(fontSize: 16),
                                        ),
                                        trailing: Text(
                                          'จำนวน : $count ${isDrink ? 'ขวด' : 'กล่อง'}',
                                          style: TextStyle(
                                              fontSize: 16,
                                              fontWeight: FontWeight.bold),
                                        ),
                                      );
                                    },
                                  ),
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: List.generate(
                                      (sortedItems.length / itemsPerPage)
                                          .ceil(),
                                      (index) => ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor:
                                              currentPageItems == (index + 1)
                                                  ? Colors.orange
                                                  : const Color.fromARGB(
                                                      255, 207, 201, 201),
                                        ),
                                        onPressed: () {
                                          setState(() {
                                            currentPageItems = index + 1;
                                          });
                                        },
                                        child: Text('${index + 1}'),
                                      ),
                                    ),
                                  ),
                                ],
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  ),

                  SizedBox(height: 16),

                  // ส่วนรีวิวจากลูกค้า
                  Card(
                    elevation: 4,
                    color: Colors.white,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "รีวิวจากลูกค้า",
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.orange,
                            ),
                          ),
                          FutureBuilder<List<Map<String, dynamic>>>(
                            future:
                                fetchReviews(), // ดึงข้อมูลรีวิวจาก Firestore
                            builder: (context, snapshot) {
                              if (snapshot.connectionState ==
                                  ConnectionState.waiting) {
                                return Center(
                                    child: CircularProgressIndicator());
                              }
                              if (snapshot.hasError) {
                                return Center(
                                    child: Text(
                                        "เกิดข้อผิดพลาด: ${snapshot.error}"));
                              }

                              final List<Map<String, dynamic>> reviews =
                                  snapshot.data ?? [];

                              if (reviews.isEmpty) {
                                return Center(child: Text("ไม่มีรีวิว"));
                              }

                              return Column(
                                children: [
                                  ListView.builder(
                                    shrinkWrap: true,
                                    physics: NeverScrollableScrollPhysics(),
                                    itemCount: ((currentPageReviews - 1) *
                                                    reviewsPerPage +
                                                reviewsPerPage) >
                                            reviews.length
                                        ? reviews.length % reviewsPerPage
                                        : reviewsPerPage,
                                    itemBuilder: (context, index) {
                                      final actualIndex =
                                          (currentPageReviews - 1) *
                                                  reviewsPerPage +
                                              index;
                                      if (actualIndex >= reviews.length)
                                        return SizedBox();
                                      final review = reviews[actualIndex];

                                      return Card(
                                        elevation: 4,
                                        margin:
                                            EdgeInsets.symmetric(vertical: 5),
                                        child: Padding(
                                          padding: EdgeInsets.all(10),
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Row(
                                                children: [
                                                  Icon(Icons.account_circle,
                                                      size: 40,
                                                      color: Colors.grey),
                                                  SizedBox(width: 10),
                                                  Text(
                                                    review['name'],
                                                    style: TextStyle(
                                                        fontSize: 16,
                                                        fontWeight:
                                                            FontWeight.bold),
                                                  ),
                                                ],
                                              ),
                                              SizedBox(height: 5),
                                              Row(
                                                children: List.generate(5, (i) {
                                                  return Icon(
                                                    Icons.star,
                                                    color: i < review['star']
                                                        ? Colors.orange
                                                        : Color.fromARGB(
                                                            255, 207, 201, 201),
                                                    size: 18,
                                                  );
                                                }),
                                              ),
                                              SizedBox(height: 5),
                                              Text(review['review'],
                                                  style:
                                                      TextStyle(fontSize: 14)),
                                            ],
                                          ),
                                        ),
                                      );
                                    },
                                  ),
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: List.generate(
                                      (reviews.length / reviewsPerPage).ceil(),
                                      (index) => ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor:
                                              currentPageReviews == (index + 1)
                                                  ? Colors.orange
                                                  : Color.fromARGB(
                                                      255, 207, 201, 201),
                                        ),
                                        onPressed: () {
                                          setState(() {
                                            currentPageReviews = index + 1;
                                          });
                                        },
                                        child: Text('${index + 1}'),
                                      ),
                                    ),
                                  ),
                                ],
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  ),
                  Card(
                    elevation: 4,
                    color: Colors.white,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'ยอดขายประจำวัน:',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.orange,
                            ),
                          ),
                          SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              onPressed: () async {
                                DateTime? pickedDate = await showDatePicker(
                                  context: context,
                                  initialDate: DateTime.now(),
                                  firstDate: DateTime.now()
                                      .subtract(Duration(days: 30)),
                                  lastDate: DateTime.now(),
                                );

                                if (pickedDate != null) {
                                  setState(() {
                                    selectedDay =
                                        "${pickedDate.year}-${pickedDate.month}-${pickedDate.day}";
                                  });
                                }
                              },
                              child: Text(selectedDay ?? "เลือกวันที่"),
                            ),
                          ),
                          SizedBox(height: 10),
                          if (selectedDay != null) ...[
                            Text(
                              'ยอดขายสำหรับวันที่: $selectedDay',
                              style: TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                            Text(
                              '${dailySales[selectedDay] ?? 0.0} บาท',
                              style: TextStyle(fontSize: 16),
                            ),
                            SizedBox(height: 10),
                            Text(
                              'สถิติยอดขายเมนูอาหารประจำวัน',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.orange,
                              ),
                            ),
                            FutureBuilder<List<Map<String, dynamic>>>(
                              future: fetchFoodMenu(selectedDay!),
                              builder: (context, snapshot) {
                                if (snapshot.connectionState ==
                                    ConnectionState.waiting) {
                                  return Center(
                                      child: CircularProgressIndicator());
                                }
                                if (snapshot.hasError) {
                                  return Center(
                                    child: Text(
                                        "เกิดข้อผิดพลาด: ${snapshot.error}"),
                                  );
                                }

                                final foodMenu = snapshot.data ?? [];
                                if (foodMenu.isEmpty) {
                                  return Text(
                                    'ไม่มีเมนูอาหารในวันที่เลือก',
                                    style: TextStyle(fontSize: 16),
                                  );
                                }

                                // Generate colors for menu items
                                final colorMap = generateColors(foodMenu
                                    .map((e) => e['name'].toString())
                                    .toList());

                                // Build Pie Chart
                                return Column(
                                  children: [
                                    AspectRatio(
                                      aspectRatio: 1.3,
                                      child: PieChart(
                                        PieChartData(
                                          sections: foodMenu.map((menuItem) {
                                            final quantity =
                                                menuItem['quantity'] as int;
                                            final totalQuantity =
                                                foodMenu.fold<int>(
                                                    0,
                                                    (previousValue, element) =>
                                                        previousValue +
                                                        (element['quantity']
                                                            as int));
                                            final percentage =
                                                (quantity / totalQuantity) *
                                                    100;

                                            return PieChartSectionData(
                                              value: percentage,
                                              title:
                                                  "${percentage.toStringAsFixed(1)}%",
                                              color: colorMap[menuItem['name']],
                                              radius: 50,
                                              titleStyle: TextStyle(
                                                fontSize: 14,
                                                fontWeight: FontWeight.bold,
                                                color: Colors.white,
                                              ),
                                            );
                                          }).toList(),
                                          sectionsSpace: 2,
                                          centerSpaceRadius: 40,
                                        ),
                                      ),
                                    ),
                                    SizedBox(height: 16),
                                    // Build Legend
                                    buildLegend(colorMap),
                                  ],
                                );
                              },
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: 16),
                  Card(
                    elevation: 4,
                    color: Colors.white,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'ยอดขายประจำเดือน:',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.orange,
                            ),
                          ),
                          DropdownButton<String>(
                            value: selectedMonth,
                            hint: Text("เลือกเดือน"),
                            isExpanded: true,
                            items: monthlyKeys.map((month) {
                              return DropdownMenuItem(
                                value: month,
                                child: Text(month),
                              );
                            }).toList(),
                            onChanged: (value) {
                              setState(() {
                                selectedMonth = value;
                              });
                            },
                          ),
                          if (selectedMonth != null) ...[
                            Text(
                              'ยอดขายสำหรับเดือน: $selectedMonth',
                              style: TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                            Text(
                              '${monthlySales[selectedMonth]!.toStringAsFixed(2)} บาท',
                              style: TextStyle(fontSize: 16),
                            ),
                            SizedBox(height: 16),
                            FutureBuilder<List<Map<String, dynamic>>>(
                              future: fetchFoodMenuByMonth(
                                  selectedMonth!), // ดึงข้อมูลรายเดือน
                              builder: (context, snapshot) {
                                if (snapshot.connectionState ==
                                    ConnectionState.waiting) {
                                  return Center(
                                      child: CircularProgressIndicator());
                                }
                                if (snapshot.hasError) {
                                  return Center(
                                    child: Text(
                                        "เกิดข้อผิดพลาด: ${snapshot.error}"),
                                  );
                                }

                                final foodMenu = snapshot.data ?? [];
                                if (foodMenu.isEmpty) {
                                  return Text(
                                    'ไม่มีเมนูอาหารในเดือนที่เลือก',
                                    style: TextStyle(fontSize: 16),
                                  );
                                }

                                final colorMap = generateColors(foodMenu
                                    .map((e) => e['name'].toString())
                                    .toList());

                                return Column(
                                  children: [
                                    AspectRatio(
                                      aspectRatio: 1.3,
                                      child: PieChart(
                                        PieChartData(
                                          sections: foodMenu.map((menuItem) {
                                            final quantity =
                                                menuItem['quantity'] as int;
                                            final totalQuantity =
                                                foodMenu.fold<int>(
                                                    0,
                                                    (previousValue, element) =>
                                                        previousValue +
                                                        (element['quantity']
                                                            as int));
                                            final percentage =
                                                (quantity / totalQuantity) *
                                                    100;

                                            return PieChartSectionData(
                                              value: percentage,
                                              title:
                                                  "${percentage.toStringAsFixed(1)}%",
                                              color: colorMap[menuItem['name']],
                                              radius: 80,
                                              titleStyle: TextStyle(
                                                fontSize: 12,
                                                fontWeight: FontWeight.bold,
                                                color: Colors.white,
                                              ),
                                            );
                                          }).toList(),
                                          sectionsSpace: 2,
                                          centerSpaceRadius: 50,
                                        ),
                                      ),
                                    ),
                                    SizedBox(height: 16),
                                    buildLegend(colorMap),
                                  ],
                                );
                              },
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
