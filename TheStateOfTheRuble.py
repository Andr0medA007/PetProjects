import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def normalize_data(data):
    """Нормализация данных в диапазон [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize_data(normalized_data, min_val, max_val):
    """Обратное преобразование нормализованных данных"""
    return normalized_data * (max_val - min_val) + min_val


def initialize_weights(input_size, hidden_size, output_size):
# Инициализация весов
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

#прямое нахождение ошибок
def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    return z1, a1, z2


#обратный поиск ошибок
def backward_propagation(X, y, z1, a1, z2, W2, learning_rate):
    m = X.shape[0]

# Градиенты
    dz2 = z2 - y
    dW2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)


    dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
    dW1 = (1 / m) * np.dot(X.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return W1, b1, W2, b2


def train_neural_network(X, y, epochs, learning_rate, input_size, hidden_size, output_size):
    # Инициализация весов
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    losses = []
    for epoch in range(epochs):
        z1, a1, z2 = forward_propagation(X, W1, b1, W2, b2)

        # Вычисление потерь
        loss = np.mean((z2 - y) ** 2)
        losses.append(loss)

        # Обратное распространение
        dW1, db1, dW2, db2 = backward_propagation(X, y, z1, a1, z2, W2, learning_rate)

        # Обновление весов
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if epoch % 1000 == 0:
            print(f"Эпоха {epoch}, Потеря: {loss:.6f}")

    return W1, b1, W2, b2, losses


def get_currency_data():
    try:
        # Получаем данные за последние 30 дней
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        # URL API ЦБ РФ для курса USD/RUB
        url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
        params = {
            'date_req1': start_date.strftime('%d/%m/%Y'),
            'date_req2': end_date.strftime('%d/%m/%Y'),
            'VAL_NM_RQ': 'R01235'  # Код USD
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Парсим XML ответ
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)

        dates = []
        rates = []

        for record in root.findall('Record'):
            date = record.get('Date')
            value = record.find('Value').text
            value = float(value.replace(',', '.'))
            dates.append(datetime.strptime(date, '%d.%m.%Y'))
            rates.append(value)

        # Создаем DataFrame и сортируем по дате
        df = pd.DataFrame({'date': dates, 'rate': rates})
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        print("Создаем тестовые данные...")
        # Создаем тестовые данные
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        base_rate = 90.0
        rates = [base_rate + np.sin(i / 5) * 2 + np.random.normal(0, 0.5) for i in range(30)]
        return pd.DataFrame({'date': dates, 'rate': rates})


def prepare_data(data, window_size=3):
    rates = data['rate'].values
    # Нормализация данных
    rates_scaled, min_val, max_val = normalize_data(rates)
    rates_scaled = rates_scaled.reshape(-1, 1)

    # Создание окон данных
    X, y = [], []
    for i in range(len(rates_scaled) - window_size):
        X.append(rates_scaled[i:i + window_size].flatten())
        y.append(rates_scaled[i + window_size])

    return np.array(X), np.array(y), min_val, max_val


def predict(X, W1, b1, W2, b2):
    #Прогнозированние данных
    z1, a1, z2 = forward_propagation(X, W1, b1, W2, b2)
    return z2


def main():
    # Получение данных
    print("Получение данных о курсе рубля...")
    data = get_currency_data()
    print(f"Получено {len(data)} записей")
    print("Последние 5 значений:")
    print(data.tail())

    # Подготавление данных
    X, y, min_val, max_val = prepare_data(data, window_size=3)

    # Разделяем на обучающую и тестовую выборки
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    # Параметры нейронной сети
    input_size = 3
    hidden_size = 3
    output_size = 1
    epochs = 10000
    learning_rate = 0.01

    # Обучаем нейронную сеть
    print("\nНачало обучения...")
    W1, b1, W2, b2, losses = train_neural_network(
        X_train, y_train, epochs, learning_rate, input_size, hidden_size, output_size
    )

    # Тестируем модель на тестовых данных
    test_predictions = predict(X_test, W1, b1, W2, b2)

    # Обратное преобразование к исходному масштабу
    y_test_original = denormalize_data(y_test, min_val, max_val)
    test_predictions_original = denormalize_data(test_predictions, min_val, max_val)

    # Прогнозируем последнее значение (которое отсутствует в данных)
    last_window = X[-1].reshape(1, -1)
    next_prediction = predict(last_window, W1, b1, W2, b2)
    next_prediction_original = denormalize_data(next_prediction, min_val, max_val)

    print(f"\nПрогноз на следующий день: {next_prediction_original[0][0]:.4f}")

    # Вычисляем метрики для тестовых данных
    mse = np.mean((test_predictions_original - y_test_original) ** 2)
    mae = np.mean(np.abs(test_predictions_original - y_test_original))

    print(f"\nМетрики качества на тестовых данных:")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.6f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.6f}")

    # Визуализация
    plt.figure(figsize=(15, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Функция потерь во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.grid(True)

    # График прогнозов
    plt.subplot(1, 2, 2)

    # Фактические данные (все)
    plt.plot(data['date'], data['rate'], label='Фактические значения', color='blue', marker='o')

    # Прогнозы на тестовых данных
    test_dates = data['date'].values[-(len(y_test)):]
    plt.plot(test_dates, test_predictions_original, label='Прогнозы на тест', color='green', marker='s')

    # Прогноз на следующий день
    next_date = data['date'].iloc[-1] + timedelta(days=1)
    plt.plot([next_date], [next_prediction_original[0][0]], 'ro', markersize=10,
             label=f'Прогноз на след. день: {next_prediction_original[0][0]:.4f}')

    # Добавляем стрелку к прогнозу для наглядности
    plt.annotate('Прогноз', xy=(next_date, next_prediction_original[0][0]),
                 xytext=(next_date - timedelta(days=2), next_prediction_original[0][0] + 1),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=12, color='red')

    plt.title('Прогноз курса рубля USD/RUB')
    plt.xlabel('Дата')
    plt.ylabel('Курс USD/RUB')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Вывод последних значений и прогноза
    print(f"\nПоследние известные значения:")
    for i in range(min(5, len(data))):
        print(f"{data['date'].iloc[-(i + 1)]}: {data['rate'].iloc[-(i + 1)]:.4f}")

    print(f"\nПрогноз на следующий день ({next_date.strftime('%Y-%m-%d')}): {next_prediction_original[0][0]:.4f}")

    # Дополнительно: прогнозируем несколько дней вперед
    print(f"\nПрогноз на несколько дней вперед:")
    current_window = last_window.copy()
    for day in range(1, 4):  # Прогноз на 3 дня вперед
        prediction = predict(current_window, W1, b1, W2, b2)
        prediction_original = denormalize_data(prediction, min_val, max_val)
        future_date = data['date'].iloc[-1] + timedelta(days=day)
        print(f"День +{day} ({future_date.strftime('%Y-%m-%d')}): {prediction_original[0][0]:.4f}")

        # Обновляем окно для следующего прогноза (сдвигаем окно)
        current_window = np.roll(current_window, -1)
        current_window[0, -1] = prediction[0][0]


if __name__ == "__main__":
    main()