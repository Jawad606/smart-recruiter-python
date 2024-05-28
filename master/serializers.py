# import serializers from the REST framework
from rest_framework import serializers
# import the todo data model
# create a serializer class


class DataSerializer(serializers.Serializer):
    filenameJob = serializers.ListField()
    contentJob = serializers.ListField()
    filenameRes = serializers.ListField()
    contentRes = serializers.ListField()
