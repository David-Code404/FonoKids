// Smoke test básico: solo confirma que la app arranca sin tirar excepciones
// al construir el widget tree (no ejercita la cámara real, que requiere un
// dispositivo físico).
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:speakshadow_app/main.dart';

void main() {
  testWidgets('La app arranca y muestra el ícono de configuración',
      (WidgetTester tester) async {
    await tester.pumpWidget(const SpeakShadowApp());
    await tester.pump();

    expect(find.byIcon(Icons.settings), findsOneWidget);
  });
}
