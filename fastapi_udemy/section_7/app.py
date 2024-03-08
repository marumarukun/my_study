import streamlit as st
import requests
import json
import datetime
import pandas as pd

page = st.sidebar.selectbox('Choose a page', ('user', 'room', 'booking'))

if page == 'user':
    
    st.title('ユーザー登録画面')

    with st.form(key='user'):
        user_name: str = st.text_input('ユーザー名', value='ぱちこり', max_chars=12)
        data = {
            'user_name': user_name
        }
        submit = st.form_submit_button('ユーザー登録')
        
    if submit:
        url = 'http://127.0.0.1:8000/users'
        response = requests.post(url, data=json.dumps(data))
        if response.status_code == 200:
            st.success('ユーザー登録成功')
        elif response.status_code == 400:
            st.error('ユーザー登録失敗, 既に登録されています')
        st.write(response.json())

if page == 'room':
    
    st.title('会議室登録画面')

    with st.form(key='room'):
        room_name: str = st.text_input('会議室名', value='ぱちこり部屋', max_chars=12)
        capacity: int = st.number_input('収容人数', min_value=1, max_value=100, value=10)
        data = {
            'room_name': room_name,
            'capacity': capacity
        }
        submit = st.form_submit_button('会議室登録')
        
    if submit:
        url = 'http://127.0.0.1:8000/rooms'
        response = requests.post(url, json=data)
        if response.status_code == 200:
            st.success('会議室登録成功')
        elif response.status_code == 400:
            st.error('会議室登録失敗, 既に登録されています')
        st.write(response.json())

if page == 'booking':
    
    st.title('会議室の予約登録画面')
    
    # ユーザー一覧を取得
    user_response = requests.get('http://127.0.0.1:8000/users')
    user_dict = {}
    for user in user_response.json():
        user_dict[user['user_name']] = user['user_id']
    user_df = pd.DataFrame(user_response.json())
    
    # 会議室一覧を取得してテーブル形式で表示
    room_response = requests.get('http://127.0.0.1:8000/rooms')
    room_dict = {}
    for room in room_response.json():
        room_dict[room['room_name']] = {
            'room_id': room['room_id'],
            'capacity': room['capacity']
        }
    room_df = pd.DataFrame(room_response.json())
    st.write('## 会議室一覧')
    st.table(room_df)
    
    # 予約一覧を取得してテーブル形式で表示
    booking_response = requests.get('http://127.0.0.1:8000/bookings')
    booking_df = pd.DataFrame(booking_response.json())
    
    booking_df = booking_df.merge(user_df, on='user_id', how='left')
    booking_df = booking_df.merge(room_df, on='room_id', how='left')
    
    # datetimeを見易くする
    func_datetime_format = lambda x: datetime.datetime.fromisoformat(x).strftime('%Y-%m-%d %H:%M')
    booking_df['start_datetime'] = booking_df['start_datetime'].apply(func_datetime_format)
    booking_df['end_datetime'] = booking_df['end_datetime'].apply(func_datetime_format)
    
    # カラムの改名と並び替え
    booking_df = booking_df.rename(columns={
        'user_name': 'ユーザー名',
        'room_name': '会議室名',
        'booked_num': '予約人数',
        'start_datetime': '開始時間',
        'end_datetime': '終了時間',
        'booking_id': '予約ID'
    })
    booking_df = booking_df[
        ['予約ID', 'ユーザー名', '会議室名', '予約人数', '開始時間', '終了時間']
        ]
    
    
    
    st.write('## 予約一覧')
    st.table(booking_df)

    with st.form(key='booking'):
        user_name: str = st.selectbox('ユーザー名', user_dict.keys())
        room_name: str = st.selectbox('会議室名', room_dict.keys())
        booked_num: int = st.number_input('予約人数', min_value=1, max_value=12, value=4)
        date: datetime.date = st.date_input('開始日', min_value=datetime.date.today(), value=datetime.date.today())
        start_time: datetime.time = st.time_input('開始時間', value=datetime.time(9, 0))
        end_time: datetime.time = st.time_input('終了時間', value=datetime.time(18, 0))
        
        submit = st.form_submit_button('予約登録')
        
    if submit:
        user_id: int = user_dict[user_name]
        room_id: int = room_dict[room_name]['room_id']
        capacity: int = room_dict[room_name]['capacity']
        
        data = {
            # 'booking_id': booking_id,
            'user_id': user_id,
            'room_id': room_id,
            'booked_num': booked_num,
            'start_datetime': datetime.datetime.combine(date, start_time).isoformat(),
            'end_datetime': datetime.datetime.combine(date, end_time).isoformat()
        }
        
        # 定員より多い予約人数の場合
        if booked_num > capacity:
            st.error(f'{room_name}の定員は、{capacity}名です。{capacity}名以下の予約人数のみ受け付けております。')
        # 開始時刻 >= 終了時刻
        elif start_time >= end_time:
            st.error('開始時刻が終了時刻を越えています')
        elif start_time < datetime.time(hour=9, minute=0, second=0) or end_time > datetime.time(hour=20, minute=0, second=0):
            st.error('利用時間は9:00~20:00になります。')
        else:
            # 会議室予約
            url = 'http://127.0.0.1:8000/bookings'
            res = requests.post(
                url,
                data=json.dumps(data)
            )
            if res.status_code == 200:
                st.success('予約完了しました')            
            elif res.status_code == 404 and res.json()['detail'] == 'Already booked' :
                st.error('指定の時間にはすでに予約が入っています。')
