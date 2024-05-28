from rest_framework import views
from rest_framework.response import Response
from rest_framework import viewsets
from .serializers import DataSerializer
from rest_framework import status
from Smart_Recuriter import fileReader


class FileUploadViewSet(viewsets.ViewSet):
    def create(self, request):
        serializer_class = DataSerializer(data=request.data)
        if not serializer_class.is_valid():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        else:
            print('pass')
            filenameJob = readData(serializer_class.data['filenameJob'])
            contentJob = readData(serializer_class.data['contentJob'])
            filenameRes = readData(serializer_class.data['filenameRes'])
            contentRes = readData(serializer_class.data['contentRes'])
            resume=convert(filenameRes,contentRes)
            job=convert(filenameJob,contentJob)
            result = fileReader.Algorithm(job, resume)
            print(result)
            return Response(result)
def readData(datas):
    value = []
    for val in datas:
        value.append(val)
    return value




def convert(filenameRes,contentRes):
    main=[]
    for i in range(len(filenameRes)):
        value=[]
        value.append(filenameRes[i])
        value.append(contentRes[i])
        main.append(value)
    return(main)
