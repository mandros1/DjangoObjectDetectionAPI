from django.contrib.auth.models import User, Group
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ObjectDetectionAPI.quickstart.serializers import GroupSerializer, UserSerializer
from ObjectDetectionAPI.quickstart.utils import parse_script


class UserViewSet(viewsets.ModelViewSet):
    """
        API endpoint that allows users to be viewed and edited
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
        API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer


@api_view(['POST'])
def image_parsing(request):
    token = request.data.__getitem__('token')
    if token:
        valid_token = settings.TOKEN
        if token == valid_token:
            image_base64 = request.data.__getitem__('imageBase64')
            # json_data = parse_image_path(image_base64)
            json_data = parse_script.parse_image_path(image_base64)
            return Response(str(json_data), status=status.HTTP_200_OK)
        else:
            return Response('Invalid token', status=status.HTTP_403_FORBIDDEN)
    else:
        return Response('Missing token', status=status.HTTP_401_UNAUTHORIZED)
